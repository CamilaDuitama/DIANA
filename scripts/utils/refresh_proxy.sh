#!/bin/bash
# Refresh the Pasteur Squid proxy credentials.
#
# Why this exists
# ---------------
# All outbound traffic on maestro goes through maestro-squid.maestro.pasteur.fr:3128.
# /etc/profile.d/squid.sh builds the proxy URL with a *munge token* as the password,
# minted with `munge -t -1` (max TTL). munged's max TTL is 3600 s and is not
# overridden here, so the credential is valid for at most one hour.
#
# That file only runs for login shells. Two consequences:
#
#   1. A long-lived process (an agent session, an editor, a shell open for hours)
#      keeps the token it inherited at startup. After an hour the proxy answers
#      407 and *every* host fails identically -- NCBI, EBI, PyPI, gitlab.pasteur.fr.
#      This looks exactly like "the sandbox has no network", and is not.
#   2. A non-login shell (`#!/bin/bash` in an sbatch script) has no proxy at all
#      unless it inherited one. A job that sat in the queue for over an hour
#      inherits an already-dead token.
#
# In both cases the cure is to mint a new token, which is what this does.
#
# Usage
# -----
#   source scripts/utils/refresh_proxy.sh        # refresh the current shell
#   scripts/utils/refresh_proxy.sh curl -sS URL  # run one command with a fresh token
#
# In an sbatch script, source it after the #SBATCH block and before any download,
# so the token is minted when the job *runs* rather than when it was submitted.

_diana_refresh_proxy() {
    if ! command -v munge >/dev/null 2>&1; then
        echo "refresh_proxy: munge not found; cannot mint proxy credentials" >&2
        return 1
    fi
    if [ ! -r /etc/profile.d/squid.sh ]; then
        echo "refresh_proxy: /etc/profile.d/squid.sh not readable" >&2
        return 1
    fi
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    # shellcheck disable=SC1091
    source /etc/profile.d/squid.sh
    if [ -z "${http_proxy:-}" ]; then
        echo "refresh_proxy: proxy still unset after sourcing squid.sh" >&2
        return 1
    fi
    return 0
}

# Sourced: export into the caller's shell. Executed: run "$@" with a fresh token.
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    _diana_refresh_proxy
else
    set -euo pipefail
    _diana_refresh_proxy || exit 1
    if [ "$#" -eq 0 ]; then
        echo "proxy refreshed; token valid for up to 1 hour" >&2
        exit 0
    fi
    exec "$@"
fi
