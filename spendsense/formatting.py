"""Indian number formatting shared by tools, alerts and the API."""


def inr(x: float, decimals: int = 0) -> str:
    """123456.5 -> '₹1,23,457' (decimals=0) or '₹1,23,456.50' (decimals=2). Negative -> '-₹...'."""
    neg = x < 0
    s = f"{abs(x):.{decimals}f}"
    whole, _, frac = s.partition(".")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups + [tail])
    return ("-" if neg else "") + "₹" + whole + (f".{frac}" if frac else "")
