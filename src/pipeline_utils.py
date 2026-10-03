"""
pipeline_utils.py

Core data loading, temporal filtering, commit deduplication, metadata parsing,
and file domain classification utilities for AIDev and MOSAIC-3M empirical research.
"""

import os
import shutil
import fnmatch
from typing import Optional, Dict, Any, List, Set, Union, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Cache directory configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "experimental")
os.makedirs(CACHE_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# Taxonomy Constants
# --------------------------------------------------------------------------
APPLICATION_EXTENSIONS: Set[str] = {
    '.py', '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.go', '.cs', '.java', '.rs', '.cpp', 
    '.c', '.rb', '.php', '.swift', '.kt', '.dart', '.vue', '.scala', '.lua', '.zig',
    '.h', '.hpp', '.html', '.css', '.sql', '.sol',
    '.erl', '.ml', '.clj', '.fs', '.ex', '.exs', '.scm', '.rkt', '.pas', '.hs', 
    '.proto', '.graphql', '.svelte', '.astro',
    '.ipynb', '.scss', '.sass', '.less', '.ejs', '.pug', '.cc', '.cxx', '.hh', '.m', '.mm',
    '.mochi', '.lean', '.mbt', '.r', '.jl', '.erb', '.cppm', '.axaml', '.xhtml', '.dm'
}

DOCS_SEGMENTS: Set[str] = {'docs', 'api_docs', 'openmetadata-docs'}

BUILD_TOOLS_SEGMENTS: Set[str] = {'scripts', 'tools', 'build', 'eng', 'bench', '.playwright'}

EXCLUDED_SEGMENTS: Set[str] = {
    'tests', 'test', '__tests__', 'spec', '__snapshots__', 
    'vendor', 'node_modules', 'venv', '.venv',
    *DOCS_SEGMENTS, *BUILD_TOOLS_SEGMENTS
}

APP_CONFIG_FILES: Set[str] = {
    'package.json', 'package-lock.json', 'pnpm-lock.yaml', 'yarn.lock', 
    'pyproject.toml', 'requirements.txt', 'cargo.toml', 'cargo.lock', 
    'pom.xml', 'build.gradle', 'tsconfig.json', 'go.mod', 'go.sum', 
    'gemfile', 'gemfile.lock',
    'composer.json', 'composer.lock', 'mix.exs', 'mix.lock', 'build.sbt',
    'makefile', 'gnumakefile', 'cmakelists.txt'
}

PURE_DOCS_EXTS: Set[str] = {
    '.md', '.mdx', '.txt', '.png', '.svg', '.rst', '.license', 
    '.out', '.bench', '.error', '.ast', '.gitignore',
    '.lock', '.dockerignore', '.npmignore', '.eslintignore', '.prettierignore', '.adoc'
}

CONFIG_EXTS: Set[str] = {
    '.json', '.yaml', '.yml', '.toml', '.ini', '.xml', '.csv', '.tsv', 
    '.cfg', '.conf', '.env', '.env.local', '.env.example', '.properties',
    '.xlf', '.config', '.po', '.strings', '.mapping', '.nix'
}

APP_DIRS: Set[str] = {
    'src', 'app', 'apps', 'frontend', 'backend', 'lib', 'libs', 
    'packages', 'sdk', 'crates', 'modules', 'pkg', 'core', 'web',
    'internal', 'compiler', 'api', 'services', 'client', 'ui', 'static', 'public',
    'transpiler', 'compile', 'examples', 'samples',
    'website', 'python', 'extensions', 'content'
}


# --------------------------------------------------------------------------
# Data Loading & Caching
# --------------------------------------------------------------------------
def load_dataset_cached(
    hf_url: str, 
    filename: str, 
    columns: Optional[List[str]] = None, 
    cache_dir: str = CACHE_DIR
) -> pd.DataFrame:
    """
    Checks if the dataset exists locally in cache_dir.
    If cached, checks schema completeness against requested columns and loads via pd.read_parquet().
    If not cached or schema is incomplete, downloads the complete table from Hugging Face,
    caches the full file to eliminate cache poisoning, and returns the requested DataFrame.
    """
    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        if columns is not None:
            existing_schema = pq.read_schema(local_path)
            missing_cols = [col for col in columns if col not in existing_schema.names]
            if not missing_cols:
                return pd.read_parquet(local_path, columns=columns)
            else:
                print(f"[Cache Stale/Incomplete] Missing columns {missing_cols}. Re-fetching full table...")
        else:
            return pd.read_parquet(local_path)
    
    parts = hf_url.replace("hf://datasets/", "").split("/", 2)
    repo_id = f"{parts[0]}/{parts[1]}"
    hf_filename = parts[2]
    
    try:
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=hf_filename, repo_type="dataset")
        shutil.copyfile(downloaded_path, local_path)
        return pd.read_parquet(local_path, columns=columns)
    except Exception:
        df = pd.read_parquet(hf_url)
        df.to_parquet(local_path, index=False)
        return pd.read_parquet(local_path, columns=columns) if columns else df


# --------------------------------------------------------------------------
# Universal Temporal Filtering
# --------------------------------------------------------------------------
def filter_temporal_cohort(
    df_pr: pd.DataFrame,
    created_start: str = "2025-06-01",
    created_end: str = "2025-07-31 23:59:59",
    merged_cutoff: str = "2025-08-31 23:59:59"
) -> pd.DataFrame:
    """
    Filters pull requests by a strict creation cohort (June 1 - July 31, 2025)
    with a 1-month merge observation window (up to August 31, 2025).
    This prevents Left-Censoring (stale PRs) and Right-Censoring (fast-merge bias).
    """
    df = df_pr.copy()
    df["merged_at_dt"] = pd.to_datetime(df["merged_at"], errors="coerce", utc=True)
    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    
    created_start_dt = pd.to_datetime(created_start, utc=True)
    created_end_dt = pd.to_datetime(created_end, utc=True)
    merged_cutoff_dt = pd.to_datetime(merged_cutoff, utc=True)
    
    mask = (
        (df["created_at_dt"] >= created_start_dt) & 
        (df["created_at_dt"] <= created_end_dt) & 
        (df["merged_at_dt"].notna()) &
        (df["merged_at_dt"] <= merged_cutoff_dt)
    )
    return df[mask].copy()


# --------------------------------------------------------------------------
# Chronological Commit Deduplication
# --------------------------------------------------------------------------
def deduplicate_commits(
    df: pd.DataFrame, 
    pr_id_col: str = "pr_id", 
    filename_col: str = "filename", 
    seq_col: str = "seq_order"
) -> pd.DataFrame:
    """
    Deduplicates commit file records per PR and filename, preserving the final commit
    status via chronological sequence ordering, but aggregating total changes.
    """
    df_out = df.copy()
    if seq_col not in df_out.columns:
        df_out[seq_col] = df_out.index
        
    # Aggregate changes across all commits for the exact same file
    if "changes" in df_out.columns:
        churn_agg = df_out.groupby([pr_id_col, filename_col])["changes"].sum().reset_index(name="cumulative_changes")
        df_out = df_out.merge(churn_agg, on=[pr_id_col, filename_col], how="left")
        df_out["changes"] = df_out["cumulative_changes"]
        df_out.drop(columns=["cumulative_changes"], inplace=True)
        
    df_out = df_out.sort_values(by=[pr_id_col, seq_col], ascending=[True, True])
    return df_out.drop_duplicates(subset=[pr_id_col, filename_col], keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------
# File Metadata Parsing
# --------------------------------------------------------------------------
def parse_file_metadata(filepath: str) -> Dict[str, str]:
    """
    Parses a file path into its constituent extension, filename, and root directory.
    """
    if not isinstance(filepath, str) or not filepath.strip():
        return {"extension": "", "filename": "", "root_dir": ""}
    
    clean_path = filepath.replace("\\", "/").strip().lstrip("/")
    fn = os.path.basename(clean_path)
    _, ext = os.path.splitext(fn)
    parts = clean_path.split("/")
    root_dir = parts[0] + "/" if len(parts) > 1 else "(root)"
    
    return {
        "extension": ext.lower() if ext else "(no extension)",
        "filename": fn,
        "root_dir": root_dir
    }


# --------------------------------------------------------------------------
# File Domain Classification
# --------------------------------------------------------------------------
def classify_file_domain(
    filepath: str, 
    status: Optional[str] = None, 
    changes: Optional[Union[int, float]] = None
) -> Dict[str, Any]:
    """
    Refactored file classifier enforcing:
    1. Unambiguous manifest priority (Tier 1: CI/CD, Containers, and IaC Manifests).
    2. Test-first priority over ambiguous scripts, application code, and configs (Tier 2).
    3. Context-dependent infrastructure (Tier 3: Automation scripts and directory-based IaC).
    4. Structural Application & Application Config classification (Tier 4).
    5. Strict mutual exclusivity and action tracking.
    """
    if not isinstance(filepath, str) or not filepath.strip():
        return {'is_app': False, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': 'Empty'}
    
    clean = filepath.replace('\\', '/').strip().lstrip('/')
    fn = os.path.basename(clean)
    _, ext = os.path.splitext(fn)
    ext = ext.lower()
    clean_lower = clean.lower()
    fn_lower = fn.lower()
    segments = [p.lower() for p in clean.split('/') if p]
    status_lower = str(status).lower() if status else ''
    is_deleted = status_lower in {'removed', 'deleted'}
    
    # --- 1. UNAMBIGUOUS INFRASTRUCTURE MANIFESTS (CI/CD, Containers, & IaC Manifests) ---
    is_infra = False
    infra_cat = None
    
    # a. CI/CD
    if ('.github/workflows/' in clean_lower or 
        '.github/actions/' in clean_lower or 
        '.circleci/' in clean_lower or 
        fnmatch.fnmatch(fn_lower, '*.gitlab-ci.yml') or 
        fnmatch.fnmatch(fn_lower, '*.gitlab-ci.yaml') or 
        fn_lower == 'jenkinsfile' or 
        fnmatch.fnmatch(fn_lower, 'azure-pipelines*.yml') or 
        fnmatch.fnmatch(fn_lower, 'azure-pipelines*.yaml') or 
        fn_lower == '.travis.yml' or 
        fn_lower == 'ci.yml'):
        is_infra = True
        infra_cat = 'CI_CD'
        
    # b. Containers
    elif (fnmatch.fnmatch(fn_lower, 'dockerfile*') or 
          fnmatch.fnmatch(fn_lower, 'containerfile*') or 
          fnmatch.fnmatch(fn_lower, 'docker-compose*.yml') or 
          fnmatch.fnmatch(fn_lower, 'docker-compose*.yaml') or 
          fn_lower in {'compose.yml', 'compose.yaml'} or 
          '/k8s/' in f'/{clean_lower}/' or 
          '/kubernetes/' in f'/{clean_lower}/' or 
          '/helm/' in f'/{clean_lower}/'):
        is_infra = True
        infra_cat = 'Containers'
        
    # c. Unambiguous Declarative & Programmatic IaC Provisioning Manifests
    elif (ext in {'.tf', '.tfvars', '.bicep'} or 
          fn_lower in {'cdk.json', 'pulumi.yaml', 'pulumi.yml', 'serverless.yml', 'serverless.yaml', 'serverless.ts'}):
        is_infra = True
        infra_cat = 'Provisioning'
        
    if is_infra:
        return {'is_app': False, 'is_infra': True, 'is_infra_deleted': is_deleted, 'infra_category': infra_cat, 'sub_type': 'Infra'}
        
    # --- 2. TEST CHECK (Guards Automation, Directory-Based IaC, and App from test fixtures) ---
    has_test_segment = any(seg in {'tests', 'test', '__tests__', 'spec', '__snapshots__'} for seg in segments[:-1])
    is_test_filename = (
        fnmatch.fnmatch(fn_lower, 'test_*.*') or 
        fnmatch.fnmatch(fn_lower, '*_test.*') or 
        fnmatch.fnmatch(fn_lower, '*.test.*') or 
        fnmatch.fnmatch(fn_lower, '*.spec.*')
    )
    if has_test_segment or is_test_filename:
        test_sub_type = 'Test_Deleted' if is_deleted else 'Test'
        return {'is_app': False, 'is_infra': False, 'is_infra_deleted': is_deleted, 'infra_category': None, 'sub_type': test_sub_type}
        
    # --- 3. CONTEXT-DEPENDENT INFRASTRUCTURE (Automation & Directory-Based IaC) ---
    if (clean_lower in {'setup.sh', 'deploy.sh'} or 
        (ext in {'.sh', '.ps1'} and any(seg in {'scripts', 'tools', 'eng', 'build'} for seg in segments))):
        is_infra = True
        infra_cat = 'Automation'
    elif (any(seg in {'terraform', 'ansible', 'pulumi', 'infrastructure', 'infra', 'provisioning'} for seg in segments[:-1]) or
          (segments[0] == 'cdk' if len(segments) > 1 else False)):
        is_infra = True
        infra_cat = 'Provisioning'
        
    if is_infra:
        return {'is_app': False, 'is_infra': True, 'is_infra_deleted': is_deleted, 'infra_category': infra_cat, 'sub_type': 'Infra'}
        
    # --- 4. REFACTORED APPLICATION & APPLICATION CONFIG CLASSIFICATION ---
    has_docs_segment = any(seg in DOCS_SEGMENTS for seg in segments[:-1])
    has_build_tools_segment = any(seg in BUILD_TOOLS_SEGMENTS for seg in segments[:-1])
    has_other_excluded_segment = has_docs_segment or has_build_tools_segment
    
    is_app = False
    app_sub_type = None

    # Rule 1: Application code or explicit top-level build/package manifests
    if (ext in APPLICATION_EXTENSIONS or ext == '.csproj' or fn_lower in APP_CONFIG_FILES) and not has_other_excluded_segment:
        is_app = True
        app_sub_type = 'App_Config' if (fn_lower in APP_CONFIG_FILES or ext == '.csproj') else 'App'
        
    # Rule 2: Config files within application directories
    elif (ext in CONFIG_EXTS or fn_lower in CONFIG_EXTS) and not has_other_excluded_segment:
        root_seg = segments[0] if segments else ''
        if root_seg in APP_DIRS:
            is_app = True
            app_sub_type = 'App_Config'
        else:
            return {'is_app': False, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': 'Generic_Config'}
            
    # Rule 3: Pure documentation or excluded directory paths
    elif has_docs_segment:
        return {'is_app': False, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': 'Docs'}
    elif has_build_tools_segment:
        return {'is_app': False, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': 'Build_Tool_Config'}
    elif ext in PURE_DOCS_EXTS or fn_lower in PURE_DOCS_EXTS:
        return {'is_app': False, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': 'Docs'}
        
    # Rule 4: Unrecognized extension fallback
    else:
        return {'is_app': False, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': 'Unrecognized_Extension'}

    if is_app:
        return {'is_app': True, 'is_infra': False, 'is_infra_deleted': False, 'infra_category': None, 'sub_type': app_sub_type}


# --------------------------------------------------------------------------
# Architectural & Ecosystem Mappings
# --------------------------------------------------------------------------
ARCHETYPE_MAPPING = {
    "src/": "Core/Library", "lib/": "Core/Library", "libs/": "Core/Library",
    "core/": "Core/Library", "packages/": "Core/Library", "pkg/": "Core/Library",
    "crates/": "Core/Library", "modules/": "Core/Library", "sdk/": "Core/Library",
    "internal/": "Core/Library", "compiler/": "Core/Library", "transpiler/": "Core/Library", "compile/": "Core/Library",
    "cmd/": "Core/Library", "source/": "Core/Library", "sources/": "Core/Library", "Sources/": "Core/Library",
    "app/": "Frontend/App", "apps/": "Frontend/App", "frontend/": "Frontend/App",
    "web/": "Frontend/App", "ui/": "Frontend/App", "client/": "Frontend/App",
    "public/": "Frontend/App", "static/": "Frontend/App",
    "backend/": "Backend/API", "api/": "Backend/API", "server/": "Backend/API", "services/": "Backend/API",
    "examples/": "Examples/Samples", "samples/": "Examples/Samples", "demo/": "Examples/Samples",
    "vendor/": "Vendor/Dependencies", "third_party/": "Vendor/Dependencies",
    "(root)": "Root (Top-Level)",
    "Cross-Domain": "Cross-Domain/Full-Stack"
}

def map_architectural_archetype(root_dir: str) -> str:
    if not isinstance(root_dir, str): return "Root/Other"
    return ARCHETYPE_MAPPING.get(root_dir.strip(), "Root/Other")

def map_ecosystem(lang) -> str:
    if pd.isna(lang): return 'Other/Niche'
    lang = str(lang).lower().strip()
    if lang in ['javascript', 'typescript', 'vue', 'svelte', 'astro', 'mdx', 'coffeescript', 'html', 'css', 'scss']: return 'Node/Frontend'
    elif lang in ['python', 'jupyter notebook']: return 'Python'
    elif lang in ['java', 'kotlin', 'scala', 'clojure', 'groovy']: return 'JVM'
    elif lang in ['c', 'c++', 'rust', 'go', 'zig', 'crystal']: return 'Compiled Systems'
    elif lang in ['c#', 'f#']: return '.NET'
    elif lang in ['ruby', 'php', 'elixir']: return 'Interpreted Backend'
    elif lang in ['swift', 'dart', 'objective-c']: return 'Mobile/Native UI'
    elif lang in ['shell', 'powershell', 'makefile', 'dockerfile', 'hcl']: return 'Ops/Scripting'
    else: return 'Other/Niche'


# --------------------------------------------------------------------------
# PR Scope Assignment
# --------------------------------------------------------------------------
def assign_pr_scope(app_count: Union[int, pd.Series, dict], infra_count: Optional[int] = None) -> str:
    """Determines the dual-scope nature of a PR based on file counts."""
    if isinstance(app_count, (pd.Series, dict)):
        row = app_count
        app_c = row.get('app_count', row.get('app_files', 0))
        infra_c = row.get('infra_count', row.get('infra_files', 0))
    else:
        app_c = app_count
        infra_c = infra_count if infra_count is not None else 0

    has_app = (app_c or 0) >= 1
    has_infra = (infra_c or 0) >= 1
    total_files = (app_c or 0) + (infra_c or 0)
    
    if has_app and has_infra and total_files >= 2: return 'App_and_Infra'
    elif has_app and not has_infra: return 'App_Only'
    elif not has_app and has_infra: return 'Infra_Only'
    else: return 'Other'


# --------------------------------------------------------------------------
# Comprehensive MOSAIC-3M File Array Parser
# --------------------------------------------------------------------------
def parse_mosaic_files_array(files_val) -> pd.Series:
    """Extracts all volume metrics and primary root directory from MOSAIC-3M nested files array."""
    if not isinstance(files_val, (list, np.ndarray)) or len(files_val) == 0:
        return pd.Series({
            "app_count": 0, "infra_count": 0, "total_files_recorded": 0, "primary_root_dir": "(none)",
            "infra_active_count": 0, "infra_deleted_count": 0, "ci_cd_count": 0, "containers_count": 0, 
            "automation_count": 0, "provisioning_count": 0, "test_count": 0, "docs_config_count": 0, 
            "trivial_app_count": 0, "unrecognized_count": 0, "has_tests": False,
            "app_line_churn": 0, "infra_line_churn": 0
        })
    
    metrics = {
        "app_count": 0, "infra_count": 0, "infra_active_count": 0, "infra_deleted_count": 0,
        "ci_cd_count": 0, "containers_count": 0, "automation_count": 0, "provisioning_count": 0,
        "test_count": 0, "docs_config_count": 0, "trivial_app_count": 0, "unrecognized_count": 0,
        "app_line_churn": 0, "infra_line_churn": 0
    }
    seen_paths = set()
    root_dir_counts = {}
    
    for f_item in files_val:
        if isinstance(f_item, dict) and "path" in f_item and f_item["path"]:
            path = f_item["path"]
            if path in seen_paths: continue
            seen_paths.add(path)
            
            file_changes = (f_item.get("additions") or 0) + (f_item.get("deletions") or 0)
            res = classify_file_domain(path, f_item.get("change_type"), file_changes)
            
            if res["is_app"]: 
                metrics["app_count"] += 1
                metrics["app_line_churn"] += file_changes
                rdir = parse_file_metadata(path)["root_dir"]
                root_dir_counts[rdir] = root_dir_counts.get(rdir, 0) + 1
            if res["is_infra"]: 
                metrics["infra_count"] += 1
                metrics["infra_line_churn"] += file_changes
                if res.get("is_infra_deleted"): metrics["infra_deleted_count"] += 1
                else: metrics["infra_active_count"] += 1
                
                cat = res.get("infra_category")
                if cat == 'CI_CD': metrics["ci_cd_count"] += 1
                elif cat == 'Containers': metrics["containers_count"] += 1
                elif cat == 'Automation': metrics["automation_count"] += 1
                elif cat == 'Provisioning': metrics["provisioning_count"] += 1
                
            stype = res.get("sub_type")
            if stype == 'Test': metrics["test_count"] += 1
            elif stype in {'Docs', 'Generic_Config', 'Docs_or_Config'}: metrics["docs_config_count"] += 1
            elif stype == 'Trivial_App': metrics["trivial_app_count"] += 1
            elif stype == 'Unrecognized_Extension': metrics["unrecognized_count"] += 1

    primary_root = "(none)"
    if root_dir_counts:
        max_val = max(root_dir_counts.values())
        max_roots = [r for r, count in root_dir_counts.items() if count == max_val]
        primary_root = "Cross-Domain" if len(max_roots) > 1 else max_roots[0]
        
    metrics["total_files_recorded"] = len(seen_paths)
    metrics["primary_root_dir"] = primary_root
    metrics["has_tests"] = metrics["test_count"] > 0
    return pd.Series(metrics)


# --------------------------------------------------------------------------
# Generalized PSM Engine
# --------------------------------------------------------------------------
def perform_greedy_strata_matching(
    df: pd.DataFrame, 
    treatment_col: str,
    treatment_val: str,
    control_val: str,
    strata_cols: list,
    distance_cols: list,
    caliper_threshold: float = 0.28,
    dataset_name: str = ""
) -> pd.DataFrame:
    """Generalized Greedy 1:1 Nearest-Neighbor Matching without replacement."""
    print(f"\n=======================================================")
    print(f"Executing Greedy 1:1 Matching: {dataset_name}")
    print(f"Treatment: {treatment_col} == '{treatment_val}' | Control: '{control_val}'")
    print(f"Strata: {strata_cols} | Distance: {distance_cols}")
    print(f"=======================================================")
    
    candidates = df[df[treatment_col].isin([treatment_val, control_val])].copy()
    matched_pairs = []
    pair_id_counter = 0
    
    grouped = candidates.groupby(strata_cols)
    for stratum_keys, stratum_df in grouped:
        treatment_units = stratum_df[stratum_df[treatment_col] == treatment_val].copy()
        treatment_units = treatment_units.sort_values(by=distance_cols[0], ascending=False)
        control_pool = stratum_df[stratum_df[treatment_col] == control_val].copy()
        
        if len(treatment_units) == 0 or len(control_pool) == 0: continue
            
        available_control_indices = set(control_pool.index)
        
        for t_idx, t_row in treatment_units.iterrows():
            if not available_control_indices: break
            
            curr_ctrls = control_pool.loc[list(available_control_indices)]
            
            # Independent calipers: filter controls where absolute diff <= caliper_threshold for EVERY distance column
            valid_ctrls = curr_ctrls
            for col in distance_cols:
                valid_ctrls = valid_ctrls[(valid_ctrls[col] - t_row[col]).abs() <= caliper_threshold]
            
            if len(valid_ctrls) == 0:
                continue
            
            # Calculate Euclidean distance only among valid candidates
            squared_diffs = np.zeros(len(valid_ctrls))
            for col in distance_cols:
                squared_diffs += (valid_ctrls[col] - t_row[col])**2
            distances = pd.Series(np.sqrt(squared_diffs), index=valid_ctrls.index)
            
            best_c_idx = distances.idxmin()
            c_row = control_pool.loc[best_c_idx]
            available_control_indices.remove(best_c_idx)
            
            pair_id_counter += 1
            
            t_record = t_row.to_dict()
            t_record["pair_id"] = pair_id_counter
            t_record["match_role"] = "Treatment"
            t_record["matching_distance"] = distances[best_c_idx]
            
            c_record = c_row.to_dict()
            c_record["pair_id"] = pair_id_counter
            c_record["match_role"] = "Control"
            c_record["matching_distance"] = distances[best_c_idx]
            
            matched_pairs.extend([t_record, c_record])
            
    df_matched = pd.DataFrame(matched_pairs)
    print(f"Successfully Matched: {len(df_matched)//2:,} pairs")
    return df_matched


# --------------------------------------------------------------------------
# Matched Cohort Generation Pipeline
# --------------------------------------------------------------------------
def generate_matched_cohorts(
    df_aidev_pr_filtered: pd.DataFrame, 
    df_mosaic_pr_filtered: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepares matching covariates (log transformations and Z-score standardizations)
    and executes greedy 1:1 nearest-neighbor matching for both AIDev v5 and MOSAIC-3M cohorts.
    
    Returns:
        tuple: (df_aidev_matched, df_mosaic_matched)
    """
    # 1. Log transformations (log_app_count, log_stars, log_app_line_churn)
    df_aidev_pr_filtered["log_app_count"] = np.log(df_aidev_pr_filtered["app_count"])
    df_mosaic_pr_filtered["log_app_count"] = np.log(df_mosaic_pr_filtered["app_count"])

    df_aidev_pr_filtered["log_stars"] = np.log(df_aidev_pr_filtered["stars"] + 1)
    df_mosaic_pr_filtered["log_stars"] = np.log(df_mosaic_pr_filtered["stars"] + 1)

    df_aidev_pr_filtered["log_app_line_churn"] = np.log(df_aidev_pr_filtered["app_line_churn"] + 1)
    df_mosaic_pr_filtered["log_app_line_churn"] = np.log(df_mosaic_pr_filtered["app_line_churn"] + 1)

    # 2. Z-score standardizations (z_log_app_count, z_log_stars, z_log_app_line_churn)
    df_aidev_pr_filtered["z_log_app_count"] = (
        df_aidev_pr_filtered["log_app_count"] - df_aidev_pr_filtered["log_app_count"].mean()
    ) / df_aidev_pr_filtered["log_app_count"].std()

    df_aidev_pr_filtered["z_log_stars"] = (
        df_aidev_pr_filtered["log_stars"] - df_aidev_pr_filtered["log_stars"].mean()
    ) / df_aidev_pr_filtered["log_stars"].std()

    df_aidev_pr_filtered["z_log_app_line_churn"] = (
        df_aidev_pr_filtered["log_app_line_churn"] - df_aidev_pr_filtered["log_app_line_churn"].mean()
    ) / df_aidev_pr_filtered["log_app_line_churn"].std()

    df_mosaic_pr_filtered["z_log_app_count"] = (
        df_mosaic_pr_filtered["log_app_count"] - df_mosaic_pr_filtered["log_app_count"].mean()
    ) / df_mosaic_pr_filtered["log_app_count"].std()

    df_mosaic_pr_filtered["z_log_stars"] = (
        df_mosaic_pr_filtered["log_stars"] - df_mosaic_pr_filtered["log_stars"].mean()
    ) / df_mosaic_pr_filtered["log_stars"].std()

    df_mosaic_pr_filtered["z_log_app_line_churn"] = (
        df_mosaic_pr_filtered["log_app_line_churn"] - df_mosaic_pr_filtered["log_app_line_churn"].mean()
    ) / df_mosaic_pr_filtered["log_app_line_churn"].std()

    # Ensure ecosystem is mapped for strata matching
    if "ecosystem" not in df_aidev_pr_filtered.columns and "primary_language" in df_aidev_pr_filtered.columns:
        df_aidev_pr_filtered["ecosystem"] = df_aidev_pr_filtered["primary_language"].apply(map_ecosystem)
    if "ecosystem" not in df_mosaic_pr_filtered.columns and "primary_language" in df_mosaic_pr_filtered.columns:
        df_mosaic_pr_filtered["ecosystem"] = df_mosaic_pr_filtered["primary_language"].apply(map_ecosystem)

    # 3. Greedy strata matching for AIDev v5
    df_aidev_matched = perform_greedy_strata_matching(
        df=df_aidev_pr_filtered,
        treatment_col="scope",
        treatment_val="App_and_Infra",
        control_val="App_Only",
        strata_cols=["agent", "architectural_archetype", "has_tests", "ecosystem"], 
        distance_cols=["z_log_app_count"],
        caliper_threshold=0.28, 
        dataset_name="AIDev v5"
    )

    # 4. Creation of df_mosaic_agentic and matching for MOSAIC-3M
    df_mosaic_agentic = df_mosaic_pr_filtered[df_mosaic_pr_filtered["agent_source"] != "Human"].copy()
    df_mosaic_matched = perform_greedy_strata_matching(
        df=df_mosaic_agentic,
        treatment_col="scope",
        treatment_val="App_and_Infra",
        control_val="App_Only",
        strata_cols=["agent_source", "architectural_archetype", "has_tests", "ecosystem"], 
        distance_cols=["z_log_app_count"],
        caliper_threshold=0.28, 
        dataset_name="MOSAIC-3M (Agentic Only)"
    )

    return df_aidev_matched, df_mosaic_matched


