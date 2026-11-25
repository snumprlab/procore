"""Pop the first queue line whose dependencies exist. Line format:
   <ENV assignments> :: <train|eval> [:: <needed file> <needed file> ...]
Called under flock by worker.sh; prints the chosen line (or nothing)."""
import os, sys
q = sys.argv[1]
lines = open(q).read().splitlines()
for i, L in enumerate(lines):
    parts = L.split(' :: ')
    needs = parts[2].split() if len(parts) > 2 else []
    if all(os.path.exists(n) for n in needs):
        open(q, 'w').write('\n'.join(lines[:i] + lines[i+1:]) + ('\n' if len(lines) > 1 else ''))
        print(L); break
