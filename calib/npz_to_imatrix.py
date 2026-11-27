import struct, sys
import numpy as np

z = np.load(sys.argv[1])
with open(sys.argv[2], "wb") as f:
    f.write(struct.pack("i", len(z.files)))
    for k in z.files:
        v = z[k].astype("<f4")
        n = k.encode()
        f.write(struct.pack("i", len(n)) + n + struct.pack("i", len(v)) + v.tobytes())
print(f"wrote {len(z.files)} tensors")
