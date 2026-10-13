# Identity Verification – Liveness Check 2.0

A browser‑based identity verification system that performs a **liveness check** using facial landmark analysis and a randomised head‑rotation challenge. The application runs entirely in the client’s browser, leveraging MediaPipe Tasks Vision and a TensorFlow Lite face‑detection model.

---

## ✨ Features

- **Real‑time face detection** – Uses the MediaPipe `FaceDetector` (TFLite model `fd_model.tflite`) to locate facial keypoints in a live webcam stream.
- **Randomised liveness challenge** – The user is asked to perform one of eight head‑movement actions (yaw / pitch combinations). The system verifies that the movement is completed with sufficient intensity.
- **Visual guidance** – A circular mask overlay and progress indicators guide the user to position their face correctly and show the progress of the liveness action.
- **Face‑size validation** – The app checks whether the face is too close or too far from the camera and prompts the user to adjust.
- **Video recording** – The session can be recorded, and the recording plus a face‑mask image can be downloaded for audit or further processing.
- **Responsive UI** – A clean interface built with HTML, CSS (Material Design components), and vanilla JavaScript.

---

## 🧠 How It Works

1. **Model initialisation** – On page load, the app loads `fd_model.tflite` via the MediaPipe `FaceDetector` and switches to video mode when the webcam starts.
2. **Webcam feed** – The user’s camera stream is displayed in a `<video>` element. The frame is processed in a `requestAnimationFrame` loop.
3. **Face validation** – For each frame, the detector returns face bounding boxes and keypoints. The app filters detections by confidence score and checks that all keypoints lie inside the inner (or outer) circle overlay.
4. **Liveness action** – A random action index (1–8) is generated. Based on the calculated yaw and pitch intensities, the app determines whether the user has performed the required head movement.
5. **Progress feedback** – As the user moves their head, the circular progress indicator fills. When the intensity reaches 1.0 for the correct action, the action is marked complete and a new action is generated.
6. **Recording & download** – The `startRecording` / `stopRecording` buttons control a `MediaRecorder`; after stopping, the video and a mask image (drawn on a canvas) can be downloaded.

---

## 🚀 Getting Started

### Prerequisites

- A modern browser that supports `getUserMedia` and WebAssembly (Chrome, Edge, Firefox, Safari).
- A webcam.
- No build step is required – the project is a static web application.

### Running the App

1. Clone the repository:
   ```bash
   git clone https://github.com/talalz94/Identity-Verification---Liveness-Check-2.0.git
   ```
2. Open `index.html` (or `video recorder.html`) directly in your browser.
   - Because the app loads local model and WASM files, you may need to serve the files via a simple HTTP server (e.g., `python -m http.server` or `npx serve`) to avoid CORS issues.
3. Allow camera access when prompted. The liveness check will begin automatically.

---

## 📁 File Structure

| File | Description |
|------|-------------|
| `index.html` / `video recorder.html` | Main entry point – contains the UI markup and loads the scripts. |
| `liveness.js` | Core logic: face detection, liveness‑action calculation, UI updates, and recording controls. |
| `style.css` | Styling and layout, including the circular mask and progress indicators. |
| `tasks-vision.js` | Bundled MediaPipe Tasks Vision library (minified). |
| `fd_model.tflite` | TensorFlow Lite face‑detection model used by MediaPipe. |
| `vision_wasm_internal.js` / `.wasm` | WebAssembly runtime files required by MediaPipe. |

---

## 🛠 Dependencies

- **MediaPipe Tasks Vision** (`@mediapipe/tasks-vision@0.10.10`) – face detection.
- **TensorFlow Lite** – model runtime (loaded via MediaPipe).
- **Material Design Components** – optional UI components referenced in `style.css`.

---

## 📄 License

The project is licensed under the **Apache License 2.0**. See the header of `style.css` for the full license notice.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open an issue or submit a pull request.

---

## 📬 Contact

For questions or feedback, please contact the repository owner [@talalz94](https://github.com/talalz94).
