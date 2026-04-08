#!/usr/bin/env bash
# =============================================================================
# Phase 13 — n8n Workflow Deploy Script
#
# Versioning convention
# ─────────────────────
# Workflows are stored in:
#   infra/n8n/workflows/v{N}/          ← versioned exports
#   infra/n8n/workflows/v{N}/CHANGELOG ← human-readable change notes
#
# Promotion path:
#   dev   → test  → prod
#   (local)  (staging) (production)
#
# Usage
# ─────
# Export a workflow from running n8n to the versioned folder:
#   ./deploy.sh export <workflow-id> <name> [version]
#
# Import all workflows from a versioned folder into n8n:
#   ./deploy.sh import <version>            # e.g. ./deploy.sh import v1
#
# Promote from test to prod (tags + git commit):
#   ./deploy.sh promote <version>
#
# List active workflows in running n8n:
#   ./deploy.sh list
#
# Environment variables required:
#   N8N_URL            — base URL of the n8n instance  (default: http://localhost:5678)
#   N8N_API_KEY        — n8n API key (Settings → API → Create API Key)
#   WORKFLOW_DIR       — directory containing versioned workflow JSONs
#                        (default: infra/n8n/workflows)
# =============================================================================

set -euo pipefail

N8N_URL="${N8N_URL:-http://localhost:5678}"
N8N_API_KEY="${N8N_API_KEY:-}"
WORKFLOW_DIR="${WORKFLOW_DIR:-$(dirname "$0")/workflows}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_require_api_key() {
  if [[ -z "${N8N_API_KEY}" ]]; then
    echo "ERROR: N8N_API_KEY is not set. Generate one in n8n Settings → API." >&2
    exit 1
  fi
}

_n8n_get() {
  local path="$1"
  curl -s -f \
    -H "X-N8N-API-KEY: ${N8N_API_KEY}" \
    -H "Accept: application/json" \
    "${N8N_URL}/api/v1${path}"
}

_n8n_post() {
  local path="$1"
  local body="$2"
  curl -s -f \
    -X POST \
    -H "X-N8N-API-KEY: ${N8N_API_KEY}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d "${body}" \
    "${N8N_URL}/api/v1${path}"
}

_n8n_put() {
  local path="$1"
  local body="$2"
  curl -s -f \
    -X PUT \
    -H "X-N8N-API-KEY: ${N8N_API_KEY}" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d "${body}" \
    "${N8N_URL}/api/v1${path}"
}

_timestamp() {
  date -u +"%Y%m%dT%H%M%SZ"
}

_log()  { echo "[$(date -u +%H:%M:%S)] $*"; }
_warn() { echo "[WARN]  $*" >&2; }
_err()  { echo "[ERROR] $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

cmd_list() {
  _require_api_key
  _log "Fetching active workflows from ${N8N_URL} …"
  _n8n_get "/workflows?active=true" | python3 -c "
import json, sys
data = json.load(sys.stdin)
workflows = data.get('data', data) if isinstance(data, dict) else data
for wf in workflows:
    status = 'ACTIVE' if wf.get('active') else 'inactive'
    print(f\"  [{status}] {wf['id']:30s} {wf['name']}\")
"
}

cmd_export() {
  local workflow_id="${1:-}"
  local name="${2:-workflow}"
  local version="${3:-v1}"

  [[ -z "${workflow_id}" ]] && _err "Usage: $0 export <workflow-id> <name> [version]"
  _require_api_key

  local out_dir="${WORKFLOW_DIR}/${version}"
  mkdir -p "${out_dir}"

  local out_file="${out_dir}/${name}.json"
  _log "Exporting workflow ${workflow_id} → ${out_file}"

  _n8n_get "/workflows/${workflow_id}" > "${out_file}"
  _log "Export complete: ${out_file}"

  # Record in CHANGELOG
  local changelog="${out_dir}/CHANGELOG"
  echo "$(_timestamp)  EXPORT  id=${workflow_id}  name=${name}  by=${USER:-unknown}" >> "${changelog}"
}

cmd_import() {
  local version="${1:-v1}"
  local env_tag="${2:-dev}"

  _require_api_key

  local src_dir="${WORKFLOW_DIR}/${version}"
  [[ -d "${src_dir}" ]] || _err "Version directory not found: ${src_dir}"

  local json_files=("${src_dir}"/*.json)
  [[ ${#json_files[@]} -eq 0 ]] && _err "No JSON files in ${src_dir}"

  _log "Importing ${#json_files[@]} workflow(s) from ${src_dir} [env=${env_tag}] …"

  local imported=0 skipped=0

  for file in "${json_files[@]}"; do
    local basename
    basename="$(basename "${file}" .json)"
    _log "  → ${basename}"

    # Inject version tag and env into the workflow JSON
    local body
    body=$(python3 -c "
import json, sys
with open('${file}') as f:
    wf = json.load(f)
wf['tags'] = list(set(wf.get('tags', []) + ['${version}', '${env_tag}']))
# Strip 'id' so n8n assigns a new one on import (avoids collision with existing)
wf.pop('id', None)
print(json.dumps(wf))
")

    # Check if workflow with this name already exists
    local existing_id
    existing_id=$(
      _n8n_get "/workflows" 2>/dev/null |
      python3 -c "
import json, sys
data = json.load(sys.stdin)
workflows = data.get('data', data) if isinstance(data, dict) else data
name = json.load(open('${file}')).get('name', '')
for wf in workflows:
    if wf.get('name') == name:
        print(wf['id'])
        break
" 2>/dev/null || true
    )

    if [[ -n "${existing_id}" ]]; then
      _log "    Updating existing workflow id=${existing_id}"
      _n8n_put "/workflows/${existing_id}" "${body}" > /dev/null
    else
      _log "    Creating new workflow"
      _n8n_post "/workflows" "${body}" > /dev/null
    fi

    (( imported++ )) || true
  done

  _log "Import done: ${imported} imported, ${skipped} skipped."

  # Record in CHANGELOG
  local changelog="${src_dir}/CHANGELOG"
  echo "$(_timestamp)  IMPORT  version=${version}  env=${env_tag}  count=${imported}  by=${USER:-unknown}" >> "${changelog}"
}

cmd_promote() {
  local version="${1:-v1}"
  local from_env="${2:-test}"
  local to_env="${3:-prod}"

  _log "Promoting ${version} from ${from_env} → ${to_env}"
  _log "Re-importing with prod env tag …"
  cmd_import "${version}" "${to_env}"

  # Tag in git if inside a git repo
  if git rev-parse --is-inside-work-tree &>/dev/null; then
    local tag="n8n-${version}-${to_env}-$(_timestamp)"
    git tag "${tag}" && _log "Git tag created: ${tag}"
    _log "Push with: git push origin ${tag}"
  else
    _warn "Not inside a git repo — skipping git tag."
  fi
}

cmd_validate() {
  local version="${1:-v1}"
  local src_dir="${WORKFLOW_DIR}/${version}"
  [[ -d "${src_dir}" ]] || _err "Version directory not found: ${src_dir}"

  _log "Validating workflow JSONs in ${src_dir} …"
  local ok=0 fail=0

  for file in "${src_dir}"/*.json; do
    local name
    name="$(basename "${file}")"
    if python3 -c "import json; json.load(open('${file}'))" 2>/dev/null; then
      _log "  ✓ ${name}"
      (( ok++ )) || true
    else
      _warn "  ✗ ${name} — invalid JSON"
      (( fail++ )) || true
    fi
  done

  _log "Validation: ${ok} OK, ${fail} failed."
  [[ ${fail} -eq 0 ]] || exit 1
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────

usage() {
  cat <<EOF
Usage: $(basename "$0") <command> [args]

Commands:
  list                          List active workflows in n8n
  export <id> <name> [version]  Export workflow from n8n to workflows/<version>/
  import <version> [env]        Import all JSONs from workflows/<version>/ into n8n
  promote <version> [from] [to] Re-import with prod tag + create git tag
  validate <version>            Validate JSON syntax of all files in version dir

Environment:
  N8N_URL        n8n base URL          (default: http://localhost:5678)
  N8N_API_KEY    n8n REST API key      (required for all commands except validate)
  WORKFLOW_DIR   workflows root dir    (default: infra/n8n/workflows)

Examples:
  N8N_API_KEY=abc123 ./deploy.sh import v1 dev
  N8N_API_KEY=abc123 ./deploy.sh promote v1 test prod
  ./deploy.sh validate v1
EOF
}

cmd="${1:-}"
shift || true

case "${cmd}" in
  list)     cmd_list ;;
  export)   cmd_export "$@" ;;
  import)   cmd_import "$@" ;;
  promote)  cmd_promote "$@" ;;
  validate) cmd_validate "$@" ;;
  *)        usage; exit 1 ;;
esac
