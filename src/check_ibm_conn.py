"""Probe IBM Quantum connectivity using the exact auth path of the real run.

Goes through q8020_cfd_qutil.backend.get_service, so a green result here means
the sweep's hardware target will authenticate identically. Resolves creds in the
same order: --token arg > IBM_QUANTUM_TOKEN env var > saved account.

Usage:
    ./.venv/bin/python src/check_ibm_conn.py
    ./.venv/bin/python src/check_ibm_conn.py --backend-name ibm_boston
    ./.venv/bin/python src/check_ibm_conn.py --instance <CRN> --token <KEY>
"""

import argparse
import sys
from typing import Any

from q8020_cfd_qutil.backend import get_service


def main() -> int:
    p = argparse.ArgumentParser(description="IBM Quantum connectivity check")
    p.add_argument("--backend-name", default=None,
                   help="Resolve this specific backend (e.g. ibm_boston).")
    p.add_argument("--token", default=None)
    p.add_argument("--channel", default="ibm_cloud")
    p.add_argument("--instance", default=None)
    args = p.parse_args()

    print(f"[check] channel={args.channel} instance={args.instance} "
          f"token={'(passed)' if args.token else '(env/saved)'}")

    try:
        service = get_service(token=args.token, channel=args.channel,
                              instance=args.instance)
    except Exception as e:
        print(f"[check] FAILED to connect: {e}", file=sys.stderr)
        return 1

    print("[check] connected.")
    try:
        acct = service.active_account() or {}
        print(f"[check] active instance: {acct.get('instance')}")
    except Exception as e:
        print(f"[check] (active_account unavailable: {e})")

    try:
        backends = service.backends(operational=True, simulator=False)
        names = [str(b.name) for b in backends]
        print(f"[check] {len(names)} operational hardware backend(s) visible:")
        for n in sorted(names):
            print(f"          - {n}")
    except Exception as e:
        print(f"[check] could not list backends: {e}", file=sys.stderr)
        return 1

    if args.backend_name:
        try:
            b: Any = service.backend(args.backend_name)
            status = b.status()
            print(f"[check] '{args.backend_name}' resolved: "
                  f"qubits={b.num_qubits} "
                  f"queue={getattr(status, 'pending_jobs', '?')} "
                  f"operational={getattr(status, 'operational', '?')}")
        except Exception as e:
            print(f"[check] '{args.backend_name}' NOT available: {e}",
                  file=sys.stderr)
            return 1

    print("[check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
