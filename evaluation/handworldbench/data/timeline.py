"""Monotonic mappings from reference time to evaluated video time."""
import bisect
import math

class Timeline:
    def __init__(self, binding):
        self.binding = binding
        kind = binding["kind"]
        if kind == "identity":
            self.x = self.y = None
        elif kind == "affine":
            if not math.isfinite(binding["scale"]) or binding["scale"] <= 0 or not math.isfinite(binding.get("offset", 0)):
                raise ValueError("Invalid affine time mapping")
        elif kind == "points":
            points = binding["points"]
            self.x, self.y = [float(p[0]) for p in points], [float(p[1]) for p in points]
            if len(points) < 2 or any(not math.isfinite(v) for v in self.x + self.y) or any(b <= a for values in (self.x, self.y) for a, b in zip(values, values[1:])):
                raise ValueError("Time mapping points must increase strictly")
        else:
            raise ValueError("Unknown time mapping")

    def at(self, t):
        kind = self.binding["kind"]
        if kind == "identity":
            return float(t)
        if kind == "affine":
            return float(t) * self.binding["scale"] + self.binding.get("offset", 0)
        if t < self.x[0] - 1e-6 or t > self.x[-1] + 1e-6:
            raise ValueError("Time outside mapping interval")
        i = max(0, min(len(self.x) - 2, bisect.bisect_right(self.x, t) - 1))
        return self.y[i] + (t-self.x[i]) / (self.x[i+1]-self.x[i]) * (self.y[i+1]-self.y[i])

    def inverse(self):
        kind = self.binding["kind"]
        if kind == "identity":
            return Timeline({"kind": "identity"})
        if kind == "affine":
            return Timeline({"kind": "affine", "scale": 1/self.binding["scale"], "offset": -self.binding.get("offset", 0)/self.binding["scale"]})
        return Timeline({"kind": "points", "points": [[y,x] for x,y in self.binding["points"]]})

