

import { FaceDetector, FilesetResolver } from "./tasks-vision.js";
let faceDetector;

let runningMode = "IMAGE";
let isBlurred = false;
let alertIsOpen = false;
// Initialize the object detector
const initializefaceDetector = async () => {

    const startTime = performance.now();
    const vision = await FilesetResolver.forVisionTasks("./");
    faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: `./fd_model.tflite`,
            delegate: "GPU"
        },
        runningMode: runningMode
    });
    // End measuring time
    const endTime = performance.now();
    const timeTaken = endTime - startTime;
    console.log(`Model Initialization took ${timeTaken} milliseconds`);
};
initializefaceDetector();
console.log("Intialization is complete !!")

const startRecordingButton = document.getElementById('startRecording');
const downloadMaskImageButton = document.getElementById('downloadMaskImage');
const downloadImageButton = document.getElementById('downloadImage');

const canvas_mask = document.getElementById('canvas_mask');
const ctx_mask = canvas_mask.getContext('2d');

//const videoElement = document.getElementById('videoElement');

let video = document.getElementById("videoElement");
let video_overlay = video

const liveView = document.getElementById("liveView");
const ovalOverlay = document.getElementById('ovalOverlay');

const instructionsTextBox = document.getElementById("textbox-pill");

//liveView.appendChild(ovalOverlay);

let left_eye;
let right_eye;
let nose;
let mouth;
let left_eye_tragion;
let right_eye_tragion;

// UI Variables
const innerCircleRadius = 150;
const outerCircleRadius = 200;
const degreeValues = {
    3: -22.50,
    5: 22.50,
    1: 67.50,
    6: 112.5,
    4: 157.5,
    8: 202.50,
    2: 247.50,
    7: 292.50
};


//Configurable Variables
const minFaceValidScore = 85;
const minFaceMaskScore = 70;
const defaultMinFaceAreaRatio = 0.35;
const defaultMaxFaceAreaRatio = 0.60;

//Feature Variables
let actionStart = false;
let actionRequired = 0;
let actionComplete = false;
let actionInitialize = true;
let rotationRequired = null;
let faceInside = false;

//UI Components
const progessIndicator = document.getElementById('pis11');
const progessIndicatorUser = document.getElementById('pis1');
const outerCircle = document.getElementById('progessIndicatorCircle2'); 




let enableWebcamButton;

document.addEventListener('DOMContentLoaded', function() {
    drawCircle();
});
// Check if webcam access is supported.
const hasGetUserMedia = () => { var _a; return !!((_a = navigator.mediaDevices) === null || _a === void 0 ? void 0 : _a.getUserMedia); };
// Keep a reference of all the child elements we create
// so we can remove them easilly on each render.
var children = [];
// If webcam supported, add event listener to button for when user
// wants to activate it.
if (hasGetUserMedia()) {
    enableCam();
    setInterval(checkCam, 5000);
}
else {
    console.warn("getUserMedia() is not supported by your browser");
}

function checkCam(){
    if (!video.srcObject) {
        if (!alertIsOpen){
        alert("Please enable webcam!")
        alertIsOpen = true;
        } else {
        alertIsOpen = false;
        }
        enableCam();   
    }
    return;
}



async function enableCam() {
    console.log("video stream: ", video.srcObject, video.paused)

    if (!faceDetector) {
        console.log("Face Detector is still loading. Please try again .... ");
        return;
    }
    // getUsermedia parameters
    const constraints = {
        video: true
    };
    // Activate the webcam stream.
    navigator.mediaDevices
        .getUserMedia(constraints)
        .then(function (stream) {
        video.srcObject = stream;
        video.addEventListener("loadeddata", predictWebcam);
    })
        .catch((err) => {
        console.error(err);
    });

}

let lastVideoTime = -1;
async function predictWebcam() {
    // if image mode is initialized, create a new classifier with video runningMode
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await faceDetector.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();

    if (actionStart === false){

        actionRequired = generateUniqueAction();
        rotationRequired = degreeValues[actionRequired];
        console.log('Setting up Action: ', actionRequired, rotationRequired);
        changeRotation(rotationRequired);

        // toggleVisibility(elementId, isVisible);

        actionStart = true;


    }
    // Detect faces using detectForVideo
    if (video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const detections = faceDetector.detectForVideo(video, startTimeMs)
            .detections;
        
        const validDetection = getValidFace(detections);
        toggleFaceUI(validDetection);

        if (validDetection){
            
            const faceSizeRatio = getFaceAreaRatio(validDetection, 150);
            adjustFace(faceSizeRatio);
            
            // Get face landmarks coordinates
            var noseTip = { x: (video.offsetWidth - validDetection.keypoints[2].x * video.offsetWidth - 3), y:  (validDetection.keypoints[2].y * video.offsetHeight - 3) }  // noseTip
            var mouth = { x: (video.offsetWidth - validDetection.keypoints[3].x * video.offsetWidth - 3), y:  (validDetection.keypoints[3].y * video.offsetHeight - 3) } 
            var left_eye = { x: (video.offsetWidth - validDetection.keypoints[0].x * video.offsetWidth - 3), y:  (validDetection.keypoints[0].y * video.offsetHeight - 3) }  // noseTip
            var left_eye_tragion = { x: (video.offsetWidth - validDetection.keypoints[4].x * video.offsetWidth - 3), y:  (validDetection.keypoints[4].y * video.offsetHeight - 3) }
            var right_eye = { x: (video.offsetWidth - validDetection.keypoints[1].x * video.offsetWidth - 3), y:  (validDetection.keypoints[1].y * video.offsetHeight - 3) }  // noseTip
            var right_eye_tragion = { x: (video.offsetWidth - validDetection.keypoints[5].x * video.offsetWidth - 3), y:  (validDetection.keypoints[5].y * video.offsetHeight - 3) }
            
            var yawIntensity = calculateYawIntensity(noseTip, mouth, left_eye, left_eye_tragion, right_eye, right_eye_tragion, faceSizeRatio);
            var pitchIntensity  = calculatePitchIntensity(noseTip, mouth, left_eye, left_eye_tragion, right_eye, right_eye_tragion, faceSizeRatio);

            // console.log("PITCH/RAW INTENSITY :", pitchIntensity, yawIntensity);

            let result = handleIntensities(yawIntensity, pitchIntensity);

            const { currentAction, output_intensity } = result
            //console.log('Result: ', output_intensity);

            if (currentAction === actionRequired){
                let percentage = output_intensity * (50 - 37) + 37;
                let actionSegment = document.getElementById('pis1')
                actionSegment.style.transform = `rotate(${rotationRequired}deg) skewY(-45deg)`;
                var element= document.querySelector('.progessIndicatorCircle')
                element.style.clipPath = `circle(${percentage}% at 50% 50%)`;

                if (output_intensity === 1) {
                    if (completeAction) {
                        console.log('Setting up new action');
                        actionStart = false;
                        completeActionUI(1);
                        // Delay execution for 2 seconds
                        await sleep(1500);
                        completeActionUI(0);
                    }
                }
                console.log('Subsequent code executed after 2 seconds');


    
            } else if (output_intensity === 0){
                const progessIndicatorUser = document.getElementById('pis1');
                if (progessIndicatorUser.classList.contains('indicatorAffects-User')) {
                    progessIndicatorUser.classList.remove('indicatorAffects-User');
                }
            }
            else {
                const currentRotation = degreeValues[currentAction];
                let actionSegment = document.getElementById('pis1')
                actionSegment.style.transform = `rotate(${currentRotation}deg) skewY(-45deg)`;
                var element= document.querySelector('.progessIndicatorCircle')
                element.style.clipPath = `circle(40% at 50% 50%)`;


            }
            


            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            var angleRoll = calculateAngle(noseTip, mouth);
            var angleYaw = calculateAngle(left_eye, left_eye_tragion); 
            var angleYawNOSE = calculateAngle(left_eye, noseTip); 
            var angleYawMOUTH = calculateAngle(left_eye, left_eye_tragion); 

            var angleYawNoseAdd =calculateAngle(right_eye, mouth) -  (180 - calculateAngle(left_eye, mouth)); 
            var distancePitch = calculateAngleOfIntersection(left_eye, left_eye_tragion, right_eye, right_eye_tragion); // suffers with YAW
            
            var midEyes = calculateMidpoint(left_eye, right_eye);
            var distancePitchNew = calculatePerpendicularDistance(midEyes, left_eye_tragion, right_eye_tragion);

            var angleRollNew = 180 - calculateAngle(left_eye, right_eye); // ROLL: this is good


            //console.log("PITCH :", distancePitchNew);
            
            // Example usage:
            const inputPitch = distancePitchNew; // Input value representing the head position
            const minRangeUP = 25;   // Minimum value of the input range
            const maxRangeUP = 40; // Maximum value of the input range
            const minRangeDOWN = 10;   // Minimum value of the input range
            const maxRangeDOWN = -5; // Maximum value of the input range
            var intensityPitch = 0.0;
            var adjustedPitch = inputPitch;

            // if (inputPitch > minRangeUP){
            //     adjustedPitch = inputPitch * Math.pow(0.25 / faceSizeRatio, 0.4);
            //     intensityPitch = mapIntensity(adjustedPitch, minRangeUP, maxRangeUP);
            // //    ; var sectionUp = getElementById('pis1'); // document.querySelector('.progessIndicatorSect:first-child');
            // //     var sectionDown =  getElementById('pis5'); //document.querySelector('.progessIndicatorSect:nth-child(5)')
            // //     sectionUp.style.background = `rgba(215, 0, 230, ${0.55})`;
            // //     sectionDown.style.background = `rgba(215, 0, 230, ${0})`;

            //     let percentage = intensityPitch * (50 - 37) + 37;
            //     var element= document.querySelector('.progessIndicatorCircle')
            //     element.style.clipPath = `circle(${percentage}% at 50% 50%)`;

            // } else if (inputPitch < minRangeDOWN) {
            //     adjustedPitch = inputPitch * Math.pow(0.40 / faceSizeRatio, 2);
            //     intensityPitch = (1 - mapIntensity(adjustedPitch, maxRangeDOWN, minRangeDOWN)) * -1;
            //     // var sectionUp = getElementById('pis1'); // document.querySelector('.progessIndicatorSect:first-child');
            //     // var sectionDown =  getElementById('pis5'); //document.querySelector('.progessIndicatorSect:nth-child(5)')
            //     // sectionUp.style.background = `rgba(215, 0, 230, ${0})`;
            //     // sectionDown.style.background = `rgba(215, 0, 230, ${0.55})`;

            //     let percentage = intensityPitch * -1 *(50 - 37) + 37;
            //     var element= document.querySelector('.progessIndicatorCircle')
            //     element.style.clipPath = `circle(${percentage}% at 50% 50%)`;
            // }
            

            // Calculate the standardized intensity value
            //console.log(inputPitch,  adjustedPitch, intensityPitch);



            //const pUp = distancePitchNew * Math.pow(0.25 / faceSizeRatio, 0.4);
            //const pDown = distancePitchNew * Math.pow(0.40 / faceSizeRatio, 2);
            //console.log(distancePitchNew, pUp, pDown)
            //console.log("Pitch: ", distancePitchNew, " - Adjusted Pitch Up:", p);
            //console.log("Pitch: ", distancePitchNew, " - Intensity:", intensityPitch);
            // console.log("YAW ANGLE DIFF:", angleYawNoseAdd);  

            var newYAW1 = calculateAngleOfIntersection(left_eye, right_eye, left_eye, mouth);
            var newYAW2 = calculateAngleOfIntersection(left_eye, right_eye, right_eye, mouth);
            var newYAWFINAL = newYAW1 - (180 - newYAW2); // experimental, affected with ROLL

            // console.log("YAW ANGLE DIFF:", newYAWFINAL, angleRollNew); 
                        
            // Calculate perpendicular distance
            // const distance = calculatePerpendicularDistance(left_eye, mouth, noseTip) + calculatePerpendicularDistance(right_eye, mouth, noseTip);//  YAW: this is good
            // console.log("Perpendicular Distance:", distance);
            // console.log("YAW ANGLE FINAL: ", (angleYawNoseAdd + distance ) / 2);

            // console.log("PITCH: ", distancePitch, " - YAW: ", angleYaw, " - ROLL: ", angleRoll);

           
                    } else {
            console.log('Invalid Face');
        }

        displayVideoDetections(detections);
    }
    // Call this function again to keep predicting when the browser is ready
    window.requestAnimationFrame(predictWebcam);
}

function completeActionUI(status) {
    const progressIndicator = document.getElementById('pis11');
    const progressIndicatorUser = document.getElementById('pis1');

    if (status === 1) {
        progressIndicatorUser.classList.remove('indicatorAffects-User');
        progressIndicator.classList.remove('indicatorAffects-Action');
        progressIndicator.classList.add('indicatorAffects-Complete');
    } else {
        progressIndicator.classList.remove('indicatorAffects-Complete');
    }
}

function toggleFaceUI(isFaceValid) {
    const progessIndicator = document.getElementById('pis11');
    const progessIndicatorUser = document.getElementById('pis1');
    const outerCircle = document.getElementById('progessIndicatorCircle2'); 

    if (isFaceValid) {
        // Handle outerCircle invalidFace class
        if (outerCircle.classList.contains('invalidFace')) {
            outerCircle.classList.remove('invalidFace');
            console.log('Invalid face effect removed.');
        }

        // Handle progessIndicator indicatorAffects-Action class
        if (!progessIndicator.classList.contains('indicatorAffects-Action')) {
            progessIndicator.classList.add('indicatorAffects-Action');
            console.log('Action effect applied.');
        }

        // Handle progessIndicatorUser indicatorAffects-User class
        if (!progessIndicatorUser.classList.contains('indicatorAffects-User')) {
            progessIndicatorUser.classList.add('indicatorAffects-User');
            console.log('User effect applied.');
        }
    } else {
        // Handle progessIndicatorUser indicatorAffects-User class
        if (progessIndicatorUser.classList.contains('indicatorAffects-User')) {
            progessIndicatorUser.classList.remove('indicatorAffects-User');
            console.log('User effect removed.');
        }

        // Handle progessIndicator indicatorAffects-Action class
        if (progessIndicator.classList.contains('indicatorAffects-Action')) {
            progessIndicator.classList.remove('indicatorAffects-Action');
            console.log('Action effect removed.');
        }

        // Handle outerCircle invalidFace class
        if (!outerCircle.classList.contains('invalidFace')) {
            outerCircle.classList.add('invalidFace');
            console.log('Invalid face effect applied.');
        }
    }
}


function completeAction() {
    const element = document.getElementById('pis1');
    if (element) {
        // Apply green background and transformation
        element.style.background = 'green';
        
        // Freeze for 1 second
        setTimeout(() => {
            // Restore original state after 1 second
            element.style.background = 'rgba(215, 0, 230, 0.553)';
            actionStart = false;
            return true;
        }, 1000); // 1000 milliseconds = 1 second
    } else {
        console.error('Element with class "progressIndicatorSect" not found.');
    }
}

function changeRotation(newRotation) {
    const firstChild = document.getElementById('pis11');
    firstChild.classList.toggle('indicatorAffects-Action');
    firstChild.style.transform = `rotate(${newRotation}deg) skewY(-45deg)`;
}


function generateUniqueAction() {
    let newNumber;
    do {
        newNumber = Math.floor(Math.random() * 8) + 1;
    } while (newNumber === actionRequired);
    actionRequired = newNumber;
    return newNumber;
}

function handleIntensities(yawIntensity, pitchIntensity) {

    let currentAction = 0;
    let output_intensity = 0;

    if (yawIntensity === 0 && pitchIntensity === 0) {
        // console.log("Case 0: Both yaw and pitch intensities are 0.");
    } else if (yawIntensity > 0 && pitchIntensity === 0) {
        // console.log("Case 1: Yaw intensity is between 0 and 1, pitch intensity is 0.");
        currentAction = 1;
        output_intensity = yawIntensity;

    } else if (yawIntensity < 0 && pitchIntensity === 0) {
        // console.log("Case 2: Yaw intensity is between -1 and 0, pitch intensity is 0.");
        currentAction = 2;
        output_intensity = yawIntensity;

    } else if (yawIntensity === 0 && pitchIntensity > 0) {
        // console.log("Case 3: Yaw intensity is 0, pitch intensity is between 0 and 1.");
        currentAction = 3;
        output_intensity = pitchIntensity;

    } else if (yawIntensity === 0 && pitchIntensity < 0) {
        // console.log("Case 4: Yaw intensity is 0, pitch intensity is between -1 and 0.");
        currentAction = 4;
        output_intensity = pitchIntensity;

    } else {

        let absYaw = Math.abs(yawIntensity);
        let absPitch = Math.abs(pitchIntensity);
        let weightedYaw = absYaw * 0.8;
        let weightedPitch = absPitch * 0.9;
        let ease_constant = 0.0;
        output_intensity = (weightedYaw + weightedPitch) / (1 - ease_constant);
        // console.log('OUTPUT INTENSITY SIDES: ', output_intensity, weightedYaw, weightedPitch)
        output_intensity= mapIntensity(output_intensity, 0, 1.0);
    
    if (yawIntensity > 0 && pitchIntensity > 0) {
        // console.log("Case 5: Yaw intensity is between 0 and 1, pitch intensity is between 0 and 1.");
        currentAction = 5;
    } else if (yawIntensity > 0 && pitchIntensity < 0) {
        // console.log("Case 6: Yaw intensity is between 0 and 1, pitch intensity is between -1 and 0.");
        currentAction = 6;
    } else if (yawIntensity < 0 && pitchIntensity > 0) {
        // console.log("Case 7: Yaw intensity is between -1 and 0, pitch intensity is between 0 and 1.");
        currentAction = 7;
    } else if (yawIntensity < 0 && pitchIntensity < 0) {
        // console.log("Case 8: Yaw intensity is between -1 and 0, pitch intensity is between -1 and 0.");
        currentAction = 8;
    } else {
        // console.log("Unexpected intensity values.");

    }
}
output_intensity = Math.abs(output_intensity)
    return {currentAction, output_intensity};
}

function calculateYawIntensity(noseTip, mouth, left_eye, left_eye_tragion, right_eye, right_eye_tragion, faceSizeRatio){

    const multiplier = 0.7
    const minYawRight = 20 * Math.pow((0.40 / faceSizeRatio), multiplier) * (1 - multiplier);
    const maxYawRight = 80 * Math.pow((0.40 / faceSizeRatio), multiplier) * (1 - multiplier);
    const minYawLeft = -20 * Math.pow((0.40 / faceSizeRatio), multiplier) * (1 - multiplier);
    const maxYawLeft = -65 * Math.pow((0.40 / faceSizeRatio), multiplier) * (1 - multiplier);

    const yawRaw = calculatePerpendicularDistance(left_eye, mouth, noseTip) + calculatePerpendicularDistance(right_eye, mouth, noseTip);//  YAW: this is good
    
    var yawIntensity = 0.0;
    var yawAdjusted = yawRaw;

    if (yawRaw > minYawRight){

        yawAdjusted = yawRaw * Math.pow((0.40 / faceSizeRatio), multiplier) * (1 - multiplier);
        yawIntensity = (mapIntensity(yawAdjusted, minYawRight, maxYawRight)) * -1;

    } else if (yawRaw < minYawLeft) {
        yawAdjusted = yawRaw * Math.pow((0.40 / faceSizeRatio), multiplier) * (1 - multiplier);
        yawIntensity = (1 - mapIntensity(yawAdjusted, maxYawLeft, minYawLeft));
    }

    return yawIntensity;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function calculatePitchIntensity(noseTip, mouth, left_eye, left_eye_tragion, right_eye, right_eye_tragion, faceSizeRatio){

    const minPitchUP = 25;   // Minimum value of the input range
    const maxPitchUP = 40; // Maximum value of the input range
    const minPitchDOWN = 15;   // Minimum value of the input range
    const maxPitchDOWN = -5; // Maximum value of the input range

    var midEyes = calculateMidpoint(left_eye, right_eye);
    var pitchRaw = calculatePerpendicularDistance(midEyes, left_eye_tragion, right_eye_tragion);

    var pitchIntensity = 0.0;
    var pitchAdjusted = pitchRaw;

    if (pitchRaw > minPitchUP){
        pitchAdjusted = pitchRaw * Math.pow(0.25 / faceSizeRatio, 0.4);
        pitchIntensity = mapIntensity(pitchAdjusted, minPitchUP, maxPitchUP);
    } else if (pitchRaw < minPitchDOWN) {
        pitchAdjusted = pitchRaw * Math.pow(0.40 / faceSizeRatio, 2);
        pitchIntensity = (1 - mapIntensity(pitchAdjusted, maxPitchDOWN, minPitchDOWN)) * -1;
    }

    return pitchIntensity;

}

function scaleValue(value, minIn = 0.10, maxIn = 0.60, minOut = 0.3, maxOut = 0.65) {
    // Clip the input value to the valid range
    value = Math.max(minIn, Math.min(value, maxIn));

    // Calculate the scaling factor
    const scaleFactor = (maxOut - minOut) / (maxIn - minIn);

    // Apply scaling and offset
    const scaledValue = scaleFactor * (value - minIn) + minOut;

    return scaledValue;
}

function calculateMidpoint(point1, point2) {
    return {
        x: (point1.x + point2.x) / 2,
        y: (point1.y + point2.y) / 2
    };
}

function mapIntensity(value, minRange, maxRange) {
    // Ensure the value is within the specified range
    value = Math.min(Math.max(value, minRange), maxRange);
    
    // Map the value from the original range to the standardized range [0, 1]
    return (value - minRange) / (maxRange - minRange);
}

// Function to calculate the distance between two points
function calculateDistance(point1, point2) {
    return Math.sqrt(Math.pow(point2.x - point1.x, 2) + Math.pow(point2.y - point1.y, 2));
}

// Function to calculate the perpendicular distance from a point to a line defined by two other points
function calculatePerpendicularDistance(point, linePoint1, linePoint2) {
    const lineLength = calculateDistance(linePoint1, linePoint2);
    const area = (linePoint2.x - linePoint1.x) * (point.y - linePoint1.y) - (point.x - linePoint1.x) * (linePoint2.y - linePoint1.y);
    const perpendicularDistance = area / lineLength;
    return perpendicularDistance;
}



function calculateAngle(noseTip, mouth) {
    // Calculate the angle between the line from noseTip to mouth and the X-axis
    var angle = Math.atan2(mouth.y - noseTip.y, mouth.x - noseTip.x) * (180 / Math.PI);

    // Convert the angle to be between 0 and 360 degrees
    if (angle < 0) {
        angle += 360;
    }

    return angle;
}

function calculateFaceDirection(leftEye, rightEye, noseTip, mouth, leftEyeTragion, rightEyeTragion) {
    // Calculate the midpoint between the eyes
    var eyeMidpoint = {
        x: (leftEye.x + rightEye.x) / 2,
        y: (leftEye.y + rightEye.y) / 2
    };

    // Calculate the angle between the line connecting the eyes and the line connecting the eyes to the nose tip
    var angle = Math.atan2(noseTip.y - eyeMidpoint.y, noseTip.x - eyeMidpoint.x) * (180 / Math.PI);

    // Calculate the rotation direction (left or right) and angle in degrees
    var rotationDirection = "straight";
    var rotationAngle = Math.abs(angle);

    if (angle > 15) {
        rotationDirection = "right";
    } else if (angle < -15) {
        rotationDirection = "left";
    }

    return {
        direction: rotationDirection,
        angle: rotationAngle
    };
}

function calculateAngleOfIntersection(line1Point1, line1Point2, line2Point1, line2Point2) {
    // Define vectors for each line
    var vector1 = {
        x: line1Point2.x - line1Point1.x,
        y: line1Point2.y - line1Point1.y
    };
    var vector2 = {
        x: line2Point2.x - line2Point1.x,
        y: line2Point2.y - line2Point1.y
    };

    // Calculate the dot product of the two vectors
    var dotProduct = (vector1.x * vector2.x) + (vector1.y * vector2.y);

    // Calculate the magnitude of each vector
    var magnitude1 = Math.sqrt(Math.pow(vector1.x, 2) + Math.pow(vector1.y, 2));
    var magnitude2 = Math.sqrt(Math.pow(vector2.x, 2) + Math.pow(vector2.y, 2));

    // Calculate the angle between the lines in radians
    var angleRadians = Math.acos(dotProduct / (magnitude1 * magnitude2));

    // Convert the angle to degrees
    var angleDegrees = angleRadians * (180 / Math.PI);

    return angleDegrees;
}


function adjustFace(faceSizeRatio){
    if (defaultMinFaceAreaRatio > faceSizeRatio){
        instructionsTextBox.textContent = "Move your face closer to the camera - " + String(faceSizeRatio);
    } else if (defaultMaxFaceAreaRatio < faceSizeRatio){
        instructionsTextBox.textContent = "Move your face further away from the camera - " + String(faceSizeRatio);
    } else{
        instructionsTextBox.textContent = "Perfect Ratio - " + String(faceSizeRatio);
        return true;
    }
}

function getFaceAreaRatio(detection, radius){

    const faceArea = detection.boundingBox.width * detection.boundingBox.height;

    const circleArea = Math.PI * Math.pow(radius, 2);

    const areaRatio = faceArea/circleArea;

    //console.log('Face Area: ', faceArea, circleArea, areaRatio);

    return areaRatio

}

function filterDetectionsScore(detections, n) {
    return detections.filter(detection => {
        return detection.categories[0].score >= n;
    });
}

function getValidFace(detections) {
    let validFaceCount = 0;
    let validFaceDetection = null;

    for (let detection of detections) {
        let detectionScore = detection.categories[0].score * 100;

        if (detectionScore >= minFaceMaskScore) {
            let isValidFace = true;

            for (let keypoint of detection.keypoints) {
                let y = keypoint.y * video.offsetHeight - 3;
                let x = video.offsetWidth - keypoint.x * video.offsetWidth - 3;
                
                //console.log('Face Inside: ', faceInside);
                let isFacePointInMask = faceInside
                    ? isPointInsideCircle(x, y, outerCircleRadius)
                    : isPointInsideCircle(x, y, innerCircleRadius);

                if (!isFacePointInMask) {
                    isValidFace = false;
                    faceInside = false;
                    break;
                } 
            }

            if (isValidFace) {
                validFaceCount++;
                validFaceDetection = detection;
                faceInside = true;
            }
        }
    }

    if (validFaceCount > 1) {
        instructionsTextBox.textContent = "Multiple faces detected";
        return false;
    } else if (validFaceCount === 1) {
        instructionsTextBox.textContent = "Face found";
        return validFaceDetection;
    } else if (detections.length >= 1) {
        instructionsTextBox.textContent = "Please ensure your face is properly inside the circle";
        return false;
    } else {
        instructionsTextBox.textContent = "Face not found";
        return false;
    }
}

function drawCircle(){

    console.log("Drawing Circle")


    const maskWidth = 300;
    const maskHeight = 400;
    const maskRadius = maskHeight / 2; // Assuming circular mask
    const maskCenterX = document.querySelector('.oval-overlay').clientWidth / 2;
    const maskCenterY = document.querySelector('.oval-overlay').clientHeight / 2;
    
    var canvas = document.getElementById("myCanvas");
    var parent = canvas.parentElement;

    // Resize the canvas to match the parent size
    function resizeCanvas() {
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
    }

    // Call the resize function initially
    resizeCanvas();
    var ctx = canvas.getContext("2d");

    ctx.fillStyle = 'blue';

    const x = maskCenterX - maskWidth /2;
    const y = maskCenterY - maskHeight /2;
    
    ctx.fillRect(x, y, maskWidth, maskHeight);

    canvas.style.zIndex = '9999';
}

function isPointInsideCircle(x, y, circleRadius) {

    const maskCenterX = document.querySelector('.oval-overlay').clientWidth / 2;
    const maskCenterY = document.querySelector('.oval-overlay').clientHeight / 2;

    const distance = Math.sqrt((x - maskCenterX) ** 2 + (y - maskCenterY) ** 2);
    
    //console.log('Checking Distance: ', distance, circleRadius);
    // Check if the distance is less than the radius of the circle
    return distance <= circleRadius;

    // const width = rectWidth * 2;
    // const height = rectHeight * 2;
    // const x0 = maskCenterX - width / 2;
    // const y0 = maskCenterY - height / 2;
    // const x1 = maskCenterX + width / 2;
    // const y1 = maskCenterY + height / 2;

    // console.log('X: ', x, ' - Y: ', y, ' - (x0, y0, x1, y1): ', x0, y0, x1, y1)

    // if ((x > x0 & x < x1) & (y > y0 & y < y1)){
    //         return true;

    // } else {
    //     return false;
    // }
  }

function extractDataPoints(detections){

    left_eye = [detections.keypoints[0].x, detections.keypoints[0].y]
    right_eye = [detections.keypoints[1].x, detections.keypoints[1].y]
    nose = [detections.keypoints[2].x, detections.keypoints[2].y]
    mouth = [detections.keypoints[3].x, detections.keypoints[3].y]
    left_eye_tragion = [detections.keypoints[4].x, detections.keypoints[4].y]
    right_eye_tragion = [detections.keypoints[5].x, detections.keypoints[5].y]

}

function displayVideoDetections(detections) {

    // Remove any highlighting from previous frame.
    for (let child of children) {
        liveView.removeChild(child);
    }
    children.splice(0);
    // Iterate through predictions and draw them to the live view
    for (let detection of detections) {
        const p = document.createElement("p");
        p.innerText =
        "Confidence: " +
        Math.round(parseFloat(detection.categories[0].score) * 100) +
        "% . Bounding Box Size: " +
        detection.boundingBox.width + "x" + detection.boundingBox.height + ".";
        p.style =
            "left: " +
                (video.offsetWidth -
                    detection.boundingBox.width -
                    detection.boundingBox.originX) +
                "px;" +
                "top: " +
                (detection.boundingBox.originY - 30) +
                "px; " +
                "width: " +
                (detection.boundingBox.width - 10) +
                "px;";
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style =
            "left: " +
                (video.offsetWidth -
                    detection.boundingBox.width -
                    detection.boundingBox.originX) +
                "px;" +
                "top: " +
                detection.boundingBox.originY +
                "px;" +
                "width: " +
                (detection.boundingBox.width - 10) +
                "px;" +
                "height: " +
                detection.boundingBox.height +
                "px;";
        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        // Store drawn objects in memory so they are queued to delete at next call
        children.push(highlighter);
        children.push(p);
        for (let keypoint of detection.keypoints) {
            const keypointEl = document.createElement("spam");
            keypointEl.className = "key-point";
            keypointEl.style.top = `${keypoint.y * video.offsetHeight - 3}px`;
            keypointEl.style.left = `${video.offsetWidth - keypoint.x * video.offsetWidth - 3}px`;
            liveView.appendChild(keypointEl);
            children.push(keypointEl);
        }
    }
}

// Function to save video frame as JPEG with a unique filename and automatically save to the "data" folder
async function saveVideoFrameAsJPEG() {
    console.log("Cropping Mask Area");
    try {
        cropImageData();
        const dataURL = canvas_mask.toDataURL('image/jpeg');
        const a = document.createElement('a');
        a.href = dataURL;
        a.download = 'masked_video.jpg';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        console.log('Mask Crop Image Saved')
    } catch (error) {
        console.error('Error saving Mask Crop Image:', error);
    }
}

downloadMaskImageButton.addEventListener("click", function() {
    console.log("Saving Frame for Download");
    saveVideoFrameAsJPEG();
});

function cropImageData() {
    const aspectRatio = video.videoWidth / video.videoHeight;
    const width = video.offsetWidth;
    const height = width / aspectRatio;

    canvas_mask.width = width;
    canvas_mask.height = height;

    console.log('Video Dims:', video.videoWidth, video.videoHeight)
    ctx_mask.save();
    ctx_mask.clearRect(0, 0, canvas_mask.width, canvas_mask.height);
     // Apply rotation transformation
    ctx_mask.translate(canvas_mask.width, 0);
    ctx_mask.scale(-1, 1);
    ctx_mask.ellipse(
        canvas_mask.width / 2, // x-coordinate of the center
        canvas_mask.height / 2, // y-coordinate of the center
        150, // horizontal radius
        200, // vertical radius
        0, // rotation (in radians), 0 for no rotation
        0, // start angle (in radians)
        Math.PI * 2 // end angle (in radians), drawing a complete ellipse
      );
      ctx_mask.closePath();
      ctx_mask.clip();
    ctx_mask.drawImage(video, 0, 0, canvas_mask.width, canvas_mask.height);
    ctx_mask.restore();
}

document.getElementById('stopRecording').addEventListener('click', function() {
    var fillElement = document.getElementById('progessIndicatorCircle');
    // Remove the class if it exists to restart the animation
    console.log('Trigger za animation')
    fillElement.classList.remove('animate');
    
    // Trigger reflow to restart the animation
    void fillElement.offsetWidth;
    
    // Add the class to start the animation
    fillElement.classList.add('animate');
  });

// Toggle blur effect on webcam stream
// startRecordingButton.addEventListener("click", function() {
//     if (isBlurred) {
//         video.style.filter = "none"; // Remove blur effect
//         isBlurred = false;
//         startRecordingButton.textContent = "Apply Blur";
//     } else {
//         video.style.filter = "blur(10px)"; // Apply blur effect
//         isBlurred = true;
//         startRecordingButton.textContent = "Remove Blur";
//     }
// });



// const startRecordingButton = document.getElementById('startRecording');
// const stopRecordingButton = document.getElementById('stopRecording');
// const downloadRecordingButton = document.getElementById('downloadRecording');
// let mediaRecorder;
// let recordedChunks = [];

// startRecordingButton.addEventListener('click', startRecording);
// stopRecordingButton.addEventListener('click', stopRecording);
// downloadRecordingButton.addEventListener('click', downloadRecording);

// async function startRecording() {
//     const stream = await navigator.mediaDevices.getUserMedia({ video: true });
//     videoElement.srcObject = stream;

//     mediaRecorder = new MediaRecorder(stream);
//     mediaRecorder.ondataavailable = handleDataAvailable;
//     mediaRecorder.start();
// }

// function handleDataAvailable(event) {
//     if (event.data.size > 0) {
//         recordedChunks.push(event.data);
//     }
// }

// function stopRecording() {
//     mediaRecorder.stop();
// }

// function downloadRecording() {
//     const blob = new Blob(recordedChunks, { type: 'video/webm' });
//     const url = URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     document.body.appendChild(a);
//     a.style.display = 'none';
//     a.href = url;
//     a.download = 'recording.webm';
//     a.click();
//     window.URL.revokeObjectURL(url);
// }