const video = document.getElementById("videoFeed");
const canvas = document.getElementById("canvas");
const resultText = document.getElementById("resultText");
const startButton = document.getElementById("startDetection");

// Start webcam stream
navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
        video.srcObject = stream;
    })
    .catch((error) => {
        console.error("Error accessing webcam:", error);
    });

// Capture frame and send to backend
async function captureAndDetect() {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame on canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert frame to blob
    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");

        try {
            const response = await fetch("http://127.0.0.1:5000/detect", {
                method: "POST",
                body: formData
            });

            const data = await response.json();
            resultText.innerText = data.result ? "⚠️ Violence Detected" : "✅ No Violence Detected";
        } catch (error) {
            console.error("Error sending frame:", error);
            resultText.innerText = "Error in detection.";
        }
    }, "image/jpeg");
}

// Start detection at intervals
let detectionInterval;
startButton.addEventListener("click", () => {
    if (!detectionInterval) {
        detectionInterval = setInterval(captureAndDetect, 2000); // Capture every 2 seconds
        startButton.innerText = "Detecting...";
        startButton.disabled = true;
    }
});
