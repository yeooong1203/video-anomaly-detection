# SHT 1024D GT Compatibility Check

- Public 1024D primary test total T: 8781
- Exact PNG16 segment GT path expected: `C:\Users\jplabuser\vad_merge\list\gt-SHT-segment-png16.npy`
- Exact PNG16 segment GT exists: False
- Fallback segment GT path inspected: `C:\Users\jplabuser\vad_merge\list\gt-SHT-segment.npy`
- Segment GT length inspected: 8803
- Segment GT length matches public 1024D test T: False
- PNG16 frame GT path: `C:\Users\jplabuser\vad_merge\list\gt-SHT-frame-png16.npy`
- PNG16 frame GT length: 140848
- Expected frame GT length public_test_T * 16: 140496
- Frame GT length equals public test T * 16: False
- Current PNG-16 GT can be reused: False

## Reason
- `gt-SHT-segment-png16.npy` is missing under the project root/list directory.
- Segment GT length does not match the public 1024D test total T.
- Frame GT length does not equal public 1024D test total T * 16.
- Reuse is unsafe without matching total T and verified video order.
