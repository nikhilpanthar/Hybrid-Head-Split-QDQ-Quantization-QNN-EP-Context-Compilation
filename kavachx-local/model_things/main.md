  The project implements a head-splitted architecture, decoupling distribution focal loss (DFL) anchor regression from class probabilities. This separati...
  When running the baseline model in kavachx-local, the entry point is:

    ./run.sh benchmark baseline    # For testing on test images
    # or
    ./run.sh webcam baseline       # For live webcam detection
  ──────
  ### 1. Which Scripts Actually Run?
  1. **run.sh** (The Bash orchestrator):
      • Verifies the Python virtual environment (venv).
      • Maps the argument "baseline" to the file 3class_split_qdq.onnx.
      • Dispatches the execution to either benchmark.py or run_webcam.py.
  2. **benchmark.py** (The core inference & benchmark engine):
      • Loads and runs the quantized ONNX model using ONNX Runtime.

  ──────
  ### 2. What We Are Doing in the Script & Key Engineering Practices
  Here is the exact step-by-step pipeline and the practices implemented in benchmark.py:
  #### Practice 1: Hybrid Head-Split Decoupling (DFL vs. Class Scores)

  Standard YOLOv8 models export a single monolithic output [1, 84, 8400] that bundles box regression and class probabilities. In this pipeline, the output
  heads are separated into two distinct outputs:

  • dfl_out: Shape [1, 64, 8400] (Distribution Focal Loss coordinates: 4 bounding box offsets × 16 bins).
  • scores_out: Shape [1, 3, 8400] (Classification probabilities for person, fire, smoke).
  │ Why this practice? Complex post-processing operators inside an ONNX graph often fail or perform poorly on edge accelerators (like Qualcomm Hexagon
  │ NPU). Decoupling them allows the heavy convolutional backbone to run INT8-quantized, while decoding runs efficiently on CPU.
  ──────
  #### Practice 2: Off-Graph DFL Decoding (dfl_decode())

  Rather than forcing the neural network to output final box coordinates:

  • The script takes the 64-channel tensor and reshapes it to (batch, 4, 16, 8400).
  • It executes a numerically stable Softmax:
                   x-max(x)
                  e
    Softmax(x) = ───────────
                    x-max(x)
                 ∑ e

  • Computes expectation weights (∑ probᵢ × i) to determine sub-pixel coordinate offsets.

  │ Why this practice? Performing regression decoding in NumPy avoids precision loss that typically occurs when quantizing continuous box coordinates to
  │ INT8.
  ──────
  #### Practice 3: Multi-Scale Anchor Generation (make_anchors())

  The script constructs 8,400 spatial anchor points across 3 feature pyramid strides:

  • P3 (Stride 8): 80 × 80 = 6, 400 anchors (small objects)
  • P4 (Stride 16): 40 × 40 = 1, 600 anchors (medium objects)
  • P5 (Stride 32): 20 × 20 = 400 anchors (large objects)

  The bounding box coordinates are recovered using:

    x1 = (anchor_x - delta_left) * stride
    y1 = (anchor_y - delta_top) * stride
    x2 = (anchor_x + delta_right) * stride
    y2 = (anchor_y + delta_bottom) * stride
  ──────
  #### Practice 4: Rigorous Preprocessing

  In preprocess():

  • Letterbox/Resize: Scales image to the network's input resolution (640 × 640).
  • Color space conversion: OpenCV BGR → RGB.
  • Normalization: Converts uint8 [0, 255] to float32 [0.0, 1.0].
  • Layout transposition: Converts HWC format to planar NCHW format: (1, 3, 640, 640).
  ──────
  #### Practice 5: Benchmark Best Practices (Warmup + Statistical Averaging)

  When measuring latency:

  • Warmup cycles (default 5 iterations): Executes inference dummy runs before taking measurements to eliminate cold-start overhead, cache misses, and
  memory allocation penalties.
  • Multi-run sampling (default 30 iterations): Measures latency across iterations to report statistical metrics:
      • Average latency (ms)
      • Minimum & Maximum latency
      • Throughput (FPS = 1000/latency)

  ──────
  #### Practice 6: Vectorized Filtering & NMS (Non-Maximum Suppression)

  In postprocess():

  • Confidence threshold filtering: Drops all candidate anchors below the threshold (default 0.25) before NMS, significantly reducing computation.
  • Coordinate rescale: Maps bounding boxes from the 640 × 640 model grid back to the original image dimensions (w₀, h₀).
  • cv2.dnn.NMSBoxes: Applies IoU-based Non-Maximum Suppression (IoU threshold 0.45) to eliminate overlapping duplicate boxes.
────────────────────────────────────────────────────────────
> what is the name of files that handle that model for run

  Here are the exact files that handle and run the baseline model in kavachx-local:
  ──────
  ### 1. The Model File

  • **3class_split_qdq.onnx**
      • The actual INT8 QDQ quantized YOLOv8m ONNX model file (~25 MB).
  ──────
  ### 2. The Python Scripts (Inference & Execution Logic)
  • **benchmark.py**
      • What it does:
          • Loads 3class_split_qdq.onnx into ONNX Runtime (CPUExecutionProvider).
          • Preprocesses images from test_images/.
          • Runs inference, warmup, and latency/FPS timing.
          • Decodes DFL bounding boxes, performs NMS, and saves annotated images to output/.

  • **run_webcam.py**
      • What it does:
          • Captures live frames from your Mac webcam.
          • Feeds frames continuously through 3class_split_qdq.onnx.
          • Draws real-time bounding boxes and serves a live video stream at http://localhost:8080.


  ──────
  ### 3. The Launcher & Setup Scripts

  • **run.sh**
      • The main entry point script. It maps the keyword baseline to 3class_split_qdq.onnx and triggers either benchmark.py or run_webcam.py.
  • **setup.sh**
      • Prepares the Python environment (venv) and installs required runtime libraries (onnxruntime, opencv-python-headless, numpy).

────────────────────────────────────────────────────────────
> what is bins here
▸ Thought for 10s
  The `dfl_decode` function reveals how the bin probabilities are converted back into distances. It reshapes the input tensor, applies a numerically stab...
  In YOLOv8 and this codebase, "bins" comes from a technique called DFL (Distribution Focal Loss) used to predict bounding boxes.
  ──────
  ### 1. Traditional vs. Bins (DFL) Approach
  • Traditional YOLO (Single Number):
  The model directly guesses one single number for each distance to the edge:
    Distance to edge = 3.7 pixels

  Problem: Object boundaries (especially smoke or fire) are often fuzzy, blurry, or occluded. Forcing the model to guess a single rigid number causes
  errors.

  • YOLOv8 DFL (16 Bins / Histogram):
  Instead of guessing one number, the model predicts a probability distribution across 16 discrete integer slots (bins) from 0 to 15:

    Bin Index:   0    1    2     3     4     5   ...  15
    Weight:      0    1    2     3     4     5   ...  15
    Prob (%):  [ 0%,  2%, 20%,  65%,  12%,  1%, ...   0% ]
                                   ▲
                          (Model thinks distance is ~3)

  Each "bin" is simply a bucket representing a possible distance from the anchor point to the object's edge.
  ──────
  ### 2. How the Final Coordinate is Calculated

  The final distance is calculated as the mathematical expectation (weighted average) across all 16 bins:

                     15
    Final Distance =  ∑  (Probability of bin i × i)
                     i=0
  Example:
  • Probability of Bin 2: 0.20
  • Probability of Bin 3: 0.65
  • Probability of Bin 4: 0.12

    Distance = (0.20 × 2) + (0.65 × 3) + (0.12 × 4) = 0.40 + 1.95 + 0.48 = 𝟐.𝟖𝟑 pixels

  This gives high sub-pixel precision while allowing the network to handle fuzzy borders.
  ──────
  ### 3. Why You See 64 Channels in the Code
  A bounding box has 4 sides (Left, Top, Right, Bottom):

    4 sides × 16 bins each = 𝟔𝟒 channels

  In benchmark.py:43-49:

    def dfl_decode(dfl_tensor, reg_max=16):
        b, c, num_anchors = dfl_tensor.shape  # c = 64
        tensor = dfl_tensor.reshape(b, 4, reg_max, num_anchors)  # Reshaped to 4 sides x 16 bins
        ...
        weights = np.arange(reg_max, dtype=np.float32)  # [0, 1, 2, ..., 15]
        return np.sum(prob * weights, axis=2)           # Weighted average of bins

────────────────────────────────────────────────────────────
> and how many boxes will be there  ??

▸ Thought for 8s, 580 tokens
  Considering YOLOv8 with an input image size of 640x640, the model utilizes three detection scales. Scale 1 (stride 8) yields 6,400 candidate boxes. Sca...
  The model starts with 8,400 raw candidate boxes, which are then filtered down to just the final detections (typically 0 to a few boxes).

  Here is how the numbers break down:
  ──────
  ### 1. Raw Candidate Boxes: 8,400

  For a standard 640 × 640 input image, YOLOv8 generates predictions across 3 different grid scales:
   Feature Scale           | Stride                  | Grid Dimensions        | Candidate Boxes        | Target Object Size
  -------------------------|-------------------------|------------------------|------------------------|--------------------------------------------------
   P3 (High resolution)    | 8                       | 80 × 80                | 6,400                  | Small objects (e.g., distant fire spark)
   P4 (Medium resolution)  | 16                      | 40 × 40                | 1,600                  | Medium objects (e.g., person standing)
   P5 (Low resolution)     | 32                      | 20 × 20                | 400                    | Large objects (e.g., large smoke cloud)
   Total                   |                         |                        | 8,400 boxes            | (This is the 8400 in tensor shape [1, 64, 8400])
  Every single point in these grids predicts a box.
  ──────
  ### 2. Step 1: Confidence Filtering (Down to ~10 – 50 boxes)
  Most of the 8,400 boxes represent empty background (walls, sky, floor).
  In benchmark.py:88:
    mask = max_scores >= conf_threshold  # Default: 0.25 (25%)
    boxes = boxes[mask]
  This discards anything below 25% confidence, usually dropping the count from 8,400 down to around 5 to 50 boxes.
  ──────
  ### 3. Step 2: NMS (Down to Actual Objects: 1 to 5 boxes)

  Multiple neighboring grid points will inevitably detect the same object (e.g., 6 overlapping boxes around one person).

  In benchmark.py:99:

    indices = cv2.dnn.NMSBoxes(xywh, scores_filt, conf_threshold, nms_threshold)

  NMS (Non-Maximum Suppression) removes the duplicate overlapping boxes, leaving only one final box per real object (e.g., 1 box for a Person, 1 box for
  Fire).

────────────────────────────────────────────────────────────
> oky so tell me if understood right thing is we have first fix our input image size then we do handling of boxes and then bins
▸ Thought for 11s
  Input image size is fixed via preprocessing, including resizing and normalization, which is correct. The subsequent handling order needs correction; th...
  You have the overall concept right, but with one important correction in the order:

  The bins come before the boxes, because the model doesn't output box coordinates directly—we must calculate the boxes from the bins!

  Here is the exact step-by-step flow:

    [1. Fix Input Image (640x640)]
                 │
                 ▼
    [2. Run Model Inference]
                 │
                 ▼
    [3. Decode BINS (DFL)]  ──► (Softmax & weighted average across 16 bins)
                 │
                 ▼
    [4. Create 8,400 BOXES] ──► (Bins turn into x1, y1, x2, y2 coordinates)
                 │
                 ▼
    [5. Filter & NMS]       ──► (Drop low confidence, remove duplicates)
                 │
                 ▼
    [6. Final Detections]
  ──────
  ### Step-by-Step Breakdown
  1. Fix Input Image Size (Preprocessing):
      • Resize/pad your image to exactly 640 × 640, convert to RGB, and normalize to [0.0, 1.0].
  2. Model Inference:
      • The model processes the image and outputs raw numbers:
          • Class scores: [1, 3, 8400]
          • Raw bin logits: [1, 64, 8400] (4 sides × 16 bins)
  3. Decode the Bins first (dfl_decode):
      • At this stage, you do not have boxes yet—you only have probability distributions over 16 bins for each side.
      • We run Softmax and weighted averages over the 16 bins to get the 4 boundary distances: (left, top, right, bottom).
  4. Construct the 8,400 Boxes:
      • Add those decoded distances to the anchor grid points to finally get the 8,400 box coordinates: (x1, y1, x2, y2).
  5. Handle / Clean the Boxes (Postprocessing):
      • Filter: Keep only boxes where confidence score ≥ 0.25.
      • Rescale: Scale coordinates from 640 × 640 back to your original camera/image size.
      • NMS: Remove overlapping duplicate boxes so only 1 neat box per detected object remains.


────────────────────────────────────────────────────────────
> and how model works on that

▸ Thought for 9s
  The initial 640x640 input image passes through the backbone, a series of convolutional layers and C2f modules, to extract visual features; specifically...
  Here is what happens inside the YOLOv8m model when it takes that 640 × 640 image and produces the detections:
  ──────
  ### The 3 Core Parts of the Model

    Input Image [1, 3, 640, 640]
                 │
                 ▼
    ┌─────────────────────────┐
    │       1. BACKBONE       │  ──► Extracts visual features (edges, colors, shapes)
    └────────────┬────────────┘
                 │ (3 feature maps: 80x80, 40x40, 20x20)
                 ▼
    ┌─────────────────────────┐
    │         2. NECK         │  ──► Blends fine details with big-picture context
    └────────────┬────────────┘
                 │
                 ▼
    ┌─────────────────────────┐
    │     3. HEAD (Split)     │
    │   ┌───────┐ ┌───────┐   │
    │   │ Class │ │  DFL  │   │  ──► Two separate heads:
    │   │ Head  │ │ Head  │   │      - Class: "What is it?" (Person, Fire, Smoke)
    │   └───┬───┘ └───┬───┘   │      - DFL:   "Where is it?" (Bins for coordinates)
    └───────┼─────────┼───────┘
            ▼         ▼
        [1,3,8400]  [1,64,8400]
  ──────
  ### 1. The Backbone (Visual Feature Extractor)

  The model passes the image through a stack of Convolutional and C2f layers:

  • Early layers: Look at basic pixels — edges, bright orange/red colors (fire), grey haze gradients (smoke), fabric textures.
  • Deeper layers: Combine those edges into concepts — human body silhouettes, flame flickers, smoke plumes.
  • As it goes deeper, it shrinks the spatial resolution into 3 pyramid levels:
      • Level 1: 80 × 80 (detailed view — finds small objects)
      • Level 2: 40 × 40 (medium view — finds regular objects)
      • Level 3: 20 × 20 (coarse view — sees the whole scene for large objects)

  ──────
  ### 2. The Neck (PANet / Feature Fusion)

  A small spark of fire needs high pixel resolution (80 × 80), but a large smoke cloud needs the big picture (20 × 20).

  • The Neck merges information top-down and bottom-up across all three scales so the model doesn't miss small sparks or large fires.
  ──────
  ### 3. The Head (Split Decoupled Head)

  At each of the 8, 400 grid points (80 × 80 + 40 × 40 + 20 × 20), the model branches into two independent tasks:

  1. Classification Head ("What is it?"):
      • Uses 1 × 1 convolutions followed by Sigmoid activation.
      • Outputs 3 probabilities for each grid point:


    Is it Person?  |  Is it Fire?  |  Is it Smoke?

  • Result: [1, 3, 8400]

  2. DFL Regression Head ("Where is it?"):
      • Predicts the 16 bin probabilities for each of the 4 boundaries (Left, Top, Right, Bottom).
      • Result: [1, 64, 8400]

  ──────
  ### 4. How INT8 Quantization Plays a Role

  Because this is an INT8 QDQ model:

  • All internal matrix multiplications (weights and activations) are processed using 8-bit integers instead of standard 32-bit floats.
  • This is why the model runs ~4x faster with a footprint of only ~25 MB (instead of ~99 MB), while retaining near-identical detection accuracy.

