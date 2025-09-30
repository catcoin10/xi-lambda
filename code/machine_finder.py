import os
import multiprocessing as mp
import sys
import json
from typing import Tuple, Dict

# ---------------------
# Config
# ---------------------
RESUME_FILE = ".resume"
NODE_LIMIT = 2_000        # abort if intermediate node count exceeds this
PRINT_NODE_LIMIT = 500
MAX_STEPS = 500
MAX_STEPS_FOR_ORACLE = 1500

# ---------------------
# Lambda calculus terms (printer)
# ---------------------
def term_to_str(t):
    if t[0] == 'V': return str(t[1])
    if t[0] == 'L': return f"(λ.{term_to_str(t[1])})"
    if t[0] == 'A': return f"({term_to_str(t[1])} {term_to_str(t[2])})"
    if t[0] == 'Ω': return "Ω"
    if t[0] == 'Ω_STATE': return f"[Ω? {term_to_str(t[1])} : {term_to_str(t[2])} | {term_to_str(t[3])}]"
    return str(t)

# ---------------------
# Encoding / decoding
# ---------------------
def encode_unary(n): return '1'*(n+1)+'0'

def encode_bin(term):
    if term[0]=='V': return '00'+encode_unary(term[1])
    if term[0]=='L': return '01'+encode_bin(term[1])
    if term[0]=='A': return '10'+encode_bin(term[1])+encode_bin(term[2])
    if term[0]=='Ω': return '11'
    return None

def decode_bin(bits, i=0):
    if bits.startswith('00', i):
        j = i + 2
        ones = 0
        while j < len(bits) and bits[j] == '1':
            ones += 1; j += 1
        if j >= len(bits) or bits[j] != '0':
            raise ValueError("bad unary")
        return ('V', ones-1), j+1
    if bits.startswith('01', i):
        body, j2 = decode_bin(bits, i+2)
        return ('L', body), j2
    if bits.startswith('10', i):
        left, j2 = decode_bin(bits, i+2)
        right, j3 = decode_bin(bits, j2)
        return ('A', left, right), j3
    if bits.startswith('11', i):
        return ('Ω',), i+2
    raise ValueError("bad encoding at " + str(i))

# ---------------------
# Evaluator
# ---------------------
def shift(d, term, cutoff=0):
    typ = term[0]
    if typ == 'V':
        k = term[1]
        return ('V', k + d) if k >= cutoff else ('V', k)
    if typ == 'L':
        return ('L', shift(d, term[1], cutoff + 1))
    if typ == 'A':
        return ('A', shift(d, term[1], cutoff), shift(d, term[2], cutoff))
    if typ == 'Ω': return term
    if typ == 'Ω_STATE':
        return ('Ω_STATE',
                shift(d, term[1], cutoff),
                shift(d, term[2], cutoff),
                shift(d, term[3], cutoff))
    raise ValueError("bad shift")

def subst(j, s, term):
    typ = term[0]
    if typ == 'V':
        return s if term[1] == j else term
    if typ == 'L':
        return ('L', subst(j + 1, shift(1, s), term[1]))
    if typ == 'A':
        return ('A', subst(j, s, term[1]), subst(j, s, term[2]))
    if typ == 'Ω': return term
    if typ == 'Ω_STATE':
        return ('Ω_STATE', subst(j, s, term[1]), subst(j, s, term[2]), subst(j, s, term[3]))
    raise ValueError("bad subst")

# cache a single omega to avoid repeated allocation
_OMEGA_SINGLETON = None
def make_omega():
    global _OMEGA_SINGLETON
    if _OMEGA_SINGLETON is None:
        inner = ('L', ('A', ('V', 0), ('V', 0)))
        _OMEGA_SINGLETON = ('A', inner, inner)
    return _OMEGA_SINGLETON

def step(term, max_steps_for_oracle=MAX_STEPS_FOR_ORACLE):
    if term[0] == 'Ω':
        return make_omega(), True
    if term[0] == 'Ω_STATE':
        x, y, z = term[1], term[2], term[3]
        res, done, _ = normalize(x, max_steps=max_steps_for_oracle, max_steps_for_oracle=max_steps_for_oracle)
        return (y if done else z), True
    if term[0] == 'A':
        f, a = term[1], term[2]
        f2, did = step(f, max_steps_for_oracle)
        if did:
            return ('A', f2, a), True
        if f[0] == 'L':
            return subst(0, a, f[1]), True
        if f[0] == 'A' and f[1][0] == 'A' and f[1][1][0] == 'Ω':
            x_arg = f[1][2]; y_arg = f[2]; z_arg = a
            return ('Ω_STATE', x_arg, y_arg, z_arg), True
        return term, False
    return term, False

# ---------------------
# Safe node counting with early cutoff
# ---------------------
def safe_count_nodes(term, limit=None) -> Tuple[int,bool]:
    stack = [term]
    count = 0
    while stack:
        t = stack.pop()
        count += 1
        if limit is not None and count > limit:
            return count, True
        typ = t[0]
        if typ == 'L':
            stack.append(t[1])
        elif typ == 'A':
            stack.append(t[2]); stack.append(t[1])
        elif typ == 'Ω_STATE':
            stack.append(t[3]); stack.append(t[2]); stack.append(t[1])
        # V and Ω are leaves
    return count, False

# ---------------------
# normalize with explosion detection
# returns (final_term, done_bool, steps_used)
# ---------------------
def normalize(term, max_steps=MAX_STEPS, max_steps_for_oracle=MAX_STEPS_FOR_ORACLE, node_limit=NODE_LIMIT):
    t = term
    steps = 0
    try:
        for i in range(max_steps):
            cnt, exceeded = safe_count_nodes(t, limit=node_limit)
            if exceeded:
                return t, False, steps
            t2, did = step(t, max_steps_for_oracle=max_steps_for_oracle)
            steps += 1
            if not did:
                return t2, True, steps
            t = t2
        return t, False, steps
    except MemoryError:
        return term, False, steps
    except RecursionError:
        return term, False, steps
    except Exception:
        return term, False, steps

# ---------------------
# Node counting (public)
# ---------------------
def count_nodes(term):
    cnt, _ = safe_count_nodes(term, limit=None)
    return cnt

# ---------------------
# Resume helpers (JSON)
# ---------------------
def load_resume() -> Tuple[int, int, Dict[str,int], Dict[str,int], Dict]:
    """Return (j, valid_count, node_histogram, step_histogram, meta)"""
    if not os.path.exists(RESUME_FILE):
        return 0, 0, {}, {}, {}
    try:
        with open(RESUME_FILE, "r") as f:
            data = json.load(f)
        # keys are strings in JSON; convert where necessary
        j = int(data.get("j", 0))
        valid_count = int(data.get("valid_count", 0))
        node_hist = {int(k): int(v) for k,v in data.get("node_histogram", {}).items()}
        step_hist = {int(k): int(v) for k,v in data.get("step_histogram", {}).items()}
        meta = data.get("meta", {})
        return j, valid_count, node_hist, step_hist, meta
    except Exception as e:
        print(f"[WARN] Couldn't load resume ({e}) — starting fresh.")
        return 0, 0, {}, {}, {}

def save_resume(j:int, valid_count:int, node_hist:Dict[int,int], step_hist:Dict[int,int], meta:Dict):
    # JSON requires string keys
    data = {
        "j": j,
        "valid_count": valid_count,
        "node_histogram": {str(k): v for k,v in node_hist.items()},
        "step_histogram": {str(k): v for k,v in step_hist.items()},
        "meta": meta
    }
    with open(RESUME_FILE, "w") as f:
        json.dump(data, f)

# ---------------------
# Main scanner
# ---------------------
def scan_space(N, start_valid=0, max_steps=MAX_STEPS, max_steps_for_oracle=MAX_STEPS_FOR_ORACLE,
               report_every=500, reverse=True, node_limit=NODE_LIMIT):
    limit = 1 << N
    j, valid_count, node_histogram, step_histogram, meta = load_resume()
    if j > 0:
        print(f"Resuming from j={j}, valid_count={valid_count}")
    best_size = meta.get("best_size", -1)
    best_binary = meta.get("best_binary", None)

    while j < limit:
        s_forward = format(j, f'0{N}b')
        s = s_forward[::-1] if reverse else s_forward

        # conservative skips
        if s.startswith('000'):
            j += 1; continue
        if s.startswith('01') and N < 4:
            j += 1; continue
        if s.startswith('10') and N < 6:
            j += 1; continue

        try:
            term, idx = decode_bin(s, 0)
            if idx != len(s):
                j += 1
                continue
        except Exception:
            j += 1
            continue

        valid_count += 1
        if valid_count <= start_valid:
            j += 1
            continue

        final, done, steps_used = normalize(term, max_steps=max_steps, max_steps_for_oracle=max_steps_for_oracle, node_limit=node_limit)

        if not done:
            # treat as non-halting (either exploded or ran out of steps)
            j += 1
            # periodically save even if nothing found
            if valid_count % report_every == 0:
                save_resume(j, valid_count, node_histogram, step_histogram, {"best_size": best_size, "best_binary": best_binary})
                print(f"[Progress] Valid: {valid_count},halted: {sum(node_histogram.values())} — saved resume at j={j}")
            continue

        # program halted within step budget
        halt_size = count_nodes(final)
        node_histogram[halt_size] = node_histogram.get(halt_size, 0) + 1
        step_histogram[steps_used] = step_histogram.get(steps_used, 0) + 1

        if halt_size > best_size:
            best_size = halt_size
            best_binary = s
            print(f"New best term (size {halt_size}) at valid #{valid_count}:")
            if halt_size <= PRINT_NODE_LIMIT:
                print(term_to_str(final))
            else:
                print(f"(term too large to pretty-print; node count = {halt_size})")
            print(f"Binary: {s}")

        # periodic save/report
        if valid_count % report_every == 0:
            save_resume(j, valid_count, node_histogram, step_histogram, {"best_size": best_size, "best_binary": best_binary})
            halted_total = sum(node_histogram.values())
            print(f"[Progress] Valid: {valid_count}, Halted: {halted_total} ({halted_total/valid_count:.2%}) — saved resume at j={j}")

        j += 1

    # final save
    save_resume(j, valid_count, node_histogram, step_histogram, {"best_size": best_size, "best_binary": best_binary})

    halted_total = sum(node_histogram.values())
    print(f"\nOut of {valid_count} valid programs (length {N} bits), {halted_total} halted.")
    if best_binary is not None:
        print("=== Final record ===")
        print(f"Size: {best_size}")
        print(f"Binary: {best_binary}")

    # print histograms sorted
    print("\nNode size histogram (node_size : count):")
    for size in sorted(node_histogram.keys()):
        print(f"{size} : {node_histogram[size]}")

    print("\nStep count histogram (steps : count):")
    for steps in sorted(step_histogram.keys()):
        print(f"{steps} : {step_histogram[steps]}")

    return best_size, best_binary, valid_count, node_histogram, step_histogram

# ---------------------
# Demo
# ---------------------
if __name__=="__main__":
    mp.set_start_method('fork', force=True)
    try:
        N = int(input("Enter bit length N: "))
        start_valid = int(input("Enter valid program to start from (0 to start fresh): "))
    except Exception:
        print("Invalid input; exiting."); sys.exit(1)
    scan_space(N, start_valid=start_valid, max_steps=MAX_STEPS, max_steps_for_oracle=MAX_STEPS_FOR_ORACLE,
               report_every=500, reverse=True, node_limit=NODE_LIMIT)
