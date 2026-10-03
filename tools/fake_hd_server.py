"""
Plumbing test server: same app, but the CatVTON engine is replaced by a fake that
pastes the garment into the mask area (no model download). Used to test the HD flow
end to end in a browser without a GPU.
"""
import sys, time
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from server import app as appmod
from server.catvton_engine import CatVTONEngine, PRESETS


class FakeEngine(CatVTONEngine):
    def load(self):
        self.state, self.device, self.message = "ready", "fake", "fake engine"

    def generate(self, person, garment, mask, preset="fast", seed=42, progress=None):
        p = PRESETS[preset]
        W, H = p["width"], p["height"]
        person = person.convert("RGB").resize((W, H))
        m = np.array(mask.convert("L").resize((W, H))) > 127
        ys, xs = np.where(m)
        out = person.copy()
        if len(xs):
            x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
            g = garment.convert("RGB").resize((x1 - x0 + 1, y1 - y0 + 1))
            canvas = np.array(out)
            gm = np.array(g)
            sub = m[y0:y1 + 1, x0:x1 + 1]
            canvas[y0:y1 + 1, x0:x1 + 1][sub] = gm[sub]
            out = Image.fromarray(canvas)
        for i in range(p["steps"]):
            time.sleep(0.05)
            if progress:
                progress(i + 1, p["steps"])
        return out


appmod.engine = FakeEngine()
appmod.engine.load()
if __name__ == "__main__":
    appmod.main()
