"""ANSI terminal logging for tracking agent execution and routing."""
import time

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"

def log_routing(tasks: list, requires_synthesis: bool, elapsed: str = "") -> None:
    print(f"\n{C.BOLD}{C.CYAN}🧭 ORCHESTRATOR — Routing Decision{C.RESET} {C.DIM}{elapsed}{C.RESET}")
    for i, t in enumerate(tasks, 1):
        agent = getattr(t, "agent", "?")
        focus = getattr(t, "focus", "")
        print(f"   Task {i}: {C.GREEN}{agent}{C.RESET} | Focus: {focus}")
