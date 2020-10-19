def monotonic_decreasing(x1, x2, y1, y2):
  return 0 if (y1 - y2) * (x1 - x2) <= 0 else 1

def monotonic_increasing(x1, x2, y1, y2):
  return 0 if (y1 - y2) * (x1 - x2) >= 0 else 1

def convex(x0, x1, x2, y0, y1, y2):
  assert x0 <= x1 and x1 <= x2
  if x2 == x0: return 0
  baseline_action = ((x2 - x1) * y0 + (x1 - x0) * y2) / (x2 - x0)
  if y1 >= baseline_action: return 0
  return 1