clc
clear all
format longG

%% Load Image Fol1er
imds = imageDatastore('tum-datasets');

% Inspect the first image
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);
%himage = imshow(currI);

%% Map Initialization
% Set random seed for reproducibility
rng(0);

% Create a cameraIntrinsics object to store the camera intrinsic parameters.
% The intrinsics for the dataset can be found at the following page:
% https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Note that the images in the dataset are already undistorted, hence there
% is no need to specify the distortion coefficients.
focalLength    = [535.4, 539.2];    % in units of pixels
principalPoint = [320.1, 247.6];    % in units of pixels
imageSize      = size(currI,[1 2]);  % in units of pixels
intrinsics     = cameraIntrinsics(focalLength, principalPoint, imageSize);

% Detect and extract ORB features
scaleFactor = 1.2;
numLevels   = 8;
numPoints = 1000;
[preFeatures, prePoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints); 

currFrameIdx = currFrameIdx + 1;
firstI       = currI; % Preserve the first frame 

isMapInitialized  = false;
minMatches = 100;

init_map_debug = {};

% Map initialization loop
while ~isMapInitialized && currFrameIdx < numel(imds.Files)
    currI = readimage(imds, currFrameIdx);
    
    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints); 
    
    currFrameIdx = currFrameIdx + 1;
    
    % Find putative feature matches
    indexPairs = matchFeatures(preFeatures, currFeatures, 'Unique', true, ...
        'MaxRatio', 0.9, 'MatchThreshold', 40);

    preMatchedPoints  = prePoints(indexPairs(:,1),:);
    currMatchedPoints = currPoints(indexPairs(:,2),:);
    
    % If not enough matches are found, check the next frame
    if size(indexPairs, 1) < minMatches
        continue
    end
    
    preMatchedPoints  = prePoints(indexPairs(:,1),:);
    currMatchedPoints = currPoints(indexPairs(:,2),:);
    
    % Compute homography and evaluate reconstruction
    [tformH, scoreH, inliersIdxH] = helperComputeHomography(preMatchedPoints, currMatchedPoints);
    
     % Compute fundamental matrix and evaluate reconstruction
    [tformF, scoreF, inliersIdxF] = helperComputeFundamentalMatrix(preMatchedPoints, currMatchedPoints);

    % Select the model based on a heuristic
    ratio = scoreH/(scoreH + scoreF);
    ratioThreshold = 0.45;
    if ratio > ratioThreshold
        inlierTformIdx = inliersIdxH;
        tform          = tformH;
    else
        inlierTformIdx = inliersIdxF;
        tform          = tformF;
    end
    
    % Computes the camera location up to scale. Use half of the 
    % points to reduce computation
    % relativeCameraPose
    % - returns the orientation and location of a calibrated camera
    %   relative to its previous pose
    %   prev_T_curr
    % translation is upto a scale (norm(relLoc) = 1 )
    inlierPrePoints  = preMatchedPoints(inlierTformIdx);
    inlierCurrPoints = currMatchedPoints(inlierTformIdx);
    [relOrient, relLoc, validFraction] = relativeCameraPose(tform, intrinsics, ...
        inlierPrePoints(1:2:end), inlierCurrPoints(1:2:end));
    
    % If not enough inliers are found, move to the next frame
    if validFraction < 0.9 || numel(size(relOrient))==3
        continue
    end
    
    % relPose : Transformation from current to prev
    relPose = rigid3d(relOrient, relLoc);
    minParallax = 1; % In degrees
    % default rigid3d is identity and 0 vector
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        rigid3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics, minParallax);
    
    if ~isValid
        continue
    end
    
    % Get the original index of features in the two key frames
    indexPairs = indexPairs(inlierTformIdx(inlierTriangulationIdx),:);
    
    isMapInitialized = true;    
end % End of map initialization loop

if isMapInitialized
    %close(himage.Parent.Parent); % Close the previous figure
    % Show matched features
    % hfeature = showMatchedFeatures(firstI, currI, prePoints(indexPairs(:,1)), ...
    %   currPoints(indexPairs(:, 2)), 'Montage');
else
    error('Unable to initialize map.')
end

%% Store Initial key frames and map points
% - imageviewset : stores the key frames and their attributes, 
%                  such as ORB descriptors, feature points and camera 
%                  poses, and connections between the key frames, such as 
%                  feature points matches and relative camera poses. it also 
%                  builds and updates a pose graph. The absolute camera poses 
%                  and relative camera poses of odometry edges are stored as 
%                  rigid3d objects. The relative camera poses of loop-closure 
%                  edges are stored as affine3d objects.
% - worldpointset : stores 3-D positions of the map points and the 3-D 
%                   into 2-D projection correspondences:
%                   which map points are observed in a key frame and 
%                   which key frames observe a map point.
% - helperViewDirectionAndDepth : \
%                   stores other attributes of map points, such as the mean 
%                   view direction, the representative ORB descriptors, and 
%                   the range of distance at which the map point can be observed.

% Create an empty imageviewset object to store key frames
vSetKeyFrames = imageviewset;

% Create an empty worldpointset object to store 3-D map points
mapPointSet   = worldpointset;

% Create a helperViewDirectionAndDepth object to store view direction and depth 
directionAndDepth = helperViewDirectionAndDepth(size(xyzWorldPoints, 1));

% Add the first key frame. Place the camera associated with the first 
% key frame at the origin, oriented along the Z-axis
preViewId     = 1;
vSetKeyFrames = addView(vSetKeyFrames, preViewId, rigid3d, 'Points', prePoints,...
    'Features', preFeatures.Features);

% Add the second key frame
currViewId    = 2;
vSetKeyFrames = addView(vSetKeyFrames, currViewId, relPose, 'Points', currPoints,...
    'Features', currFeatures.Features);


% Add connection between the first and the second key frame
vSetKeyFrames = addConnection(vSetKeyFrames, preViewId, currViewId, relPose, 'Matches', indexPairs);


% Add 3-D map points
[mapPointSet, newPointIdx] = addWorldPoints(mapPointSet, xyzWorldPoints);

% Add observations of the map points
preLocations  = prePoints.Location;
currLocations = currPoints.Location;
preScales     = prePoints.Scale;
currScales    = currPoints.Scale;

% Add image points corresponding to the map points in the first key frame
mapPointSet   = addCorrespondences(mapPointSet, preViewId, newPointIdx, indexPairs(:,1));

% Add image points corresponding to the map points in the second key frame
mapPointSet   = addCorrespondences(mapPointSet, currViewId, newPointIdx, indexPairs(:,2));

%% Refine and Visualize the Initial Reconstruction
% Refine the initial reconstruction using bundleAdjustment, that optimizes 
% both camera poses and world points to minimize the overall reprojection 
% errors. After the refinement, the attributes of the map points including 
% 3-D locations, view direction, and depth range are updated.

% Run full bundle adjustment on the first two key frames
tracks       = findTracks(vSetKeyFrames);
cameraPoses  = poses(vSetKeyFrames);

[refinedPoints, refinedAbsPoses] = bundleAdjustment(xyzWorldPoints, tracks, ...
    cameraPoses, intrinsics, 'FixedViewIDs', 1, ...
    'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7,...
    'RelativeTolerance', 1e-15, 'MaxIteration', 50);


% Scale the map and the camera pose using the median depth of map points
medianDepth = median(vecnorm(refinedPoints.'));
refinedPoints = refinedPoints / medianDepth;

refinedAbsPoses.AbsolutePose(currViewId).Translation = ...
    refinedAbsPoses.AbsolutePose(currViewId).Translation / medianDepth;
relPose.Translation = relPose.Translation/medianDepth;

% Update key frames with the refined poses
vSetKeyFrames = updateView(vSetKeyFrames, refinedAbsPoses);
vSetKeyFrames = updateConnection(vSetKeyFrames, preViewId, currViewId, relPose);

% Update map points with the refined positions
mapPointSet   = updateWorldPoints(mapPointSet, newPointIdx, refinedPoints);

% Update view direction and depth 
directionAndDepth = update(directionAndDepth, mapPointSet, vSetKeyFrames.Views, newPointIdx, true);

% Visualize matched features in the current frame
%close(hfeature.Parent.Parent);
%featurePlot   = helperVisualizeMatchedFeatures(currI, currPoints(indexPairs(:,2)));

% Visualize initial map points and camera trajectory
%mapPlot       = helperVisualizeMotionAndStructure(vSetKeyFrames, mapPointSet);

% Show legend
%showLegend(mapPlot);


%% Tracking
% The tracking process is performed using every frame and determines 
% when to insert a new key frame. 
% To simplify this example, we will terminate the tracking process 
% once a loop closure is found.

% ViewId of the current key frame
currKeyFrameId   = currViewId;

% ViewId of the last key frame
lastKeyFrameId   = currViewId;

% ViewId of the reference key frame that has the most co-visible 
% map points with the current key frame
refKeyFrameId    = currViewId;

% Index of the last key frame in the input image sequence
lastKeyFrameIdx  = currFrameIdx - 1; 

% Indices of all the key frames in the input image sequence
addedFramesIdx   = [1; lastKeyFrameIdx];

isLoopClosed     = false;


% Each frame is processed as follows:
% 1. ORB features are extracted for each new frame and then matched (using
%    matchFeatures), with features in the last key frame that have known
%    corresponding 3-D map points.
% 2. Estimate the camera pose with PnP algorithm using
%    estimateWorldCameraPose
% 3. Given the camera pose, project the map points observed by the last key
%    frame into the current frame and search for feature correspondences
%    using matchFeaturesInRadius.
% 4. With 3-D to 2-D correspondence in the current frame, refine the camera
%    pose by performing a motion only BA using bundleAdjustmentMotion.
% 5. Project the local map points into the current frame to search for more
%    feature correspondences and refine the camera pose again
% 6. The last step of tracking is to decide if the current frame is a new
%    key frame. If the current frame is a key frame, continue to Local
%    Mapping process. Otherwise, start Tracking for the next frame

% Main Loop
while ~isLoopClosed && currFrameIdx < numel(imds.Files)
    currI = readimage(imds, currFrameIdx);
    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints);
    
    % Track the last key frame
    % mapPointsIdx:   Indices of the map points observed in the current frame
    % featureIdx:     Indices of the corresponding feature points in the 
    %                 current frame
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrame(mapPointSet, ...
        vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics, scaleFactor);
    
    % Track the local map
    % refKeyFrameId:      ViewId of the reference key frame that has the most 
    %                     co-visible map points with the current frame
    % localKeyFrameIds:   ViewId of the connected key frames of the current frame
    [refKeyFrameId, localKeyFrameIds, currPose, mapPointsIdx, featureIdx] = ...
        helperTrackLocalMap(mapPointSet, directionAndDepth, vSetKeyFrames, mapPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, numLevels);

    % Check if the current frame is a key frame. 
    % A frame is a key frame if both of the following conditions are satisfied:
    %
    % 1. At least 20 frames have passed since the last key frame or the 
    %    current frame tracks fewer than 100 map points
    % 2. The map points tracked by the current frame are fewer than 90% of 
    %    points tracked by the reference key frame
    isKeyFrame = helperIsKeyFrame(mapPointSet, refKeyFrameId, lastKeyFrameIdx, ...
        currFrameIdx, mapPointsIdx);
    
    % Visualize matched features
    % updatePlot(featurePlot, currI, currPoints(featureIdx));
    
    if ~isKeyFrame
        currFrameIdx = currFrameIdx + 1;
        continue
    end
    
    % Update current key frame ID
    currKeyFrameId  = currKeyFrameId + 1;
    
    %% Local Mapping
    % Add the new key frame 
    [mapPointSet, vSetKeyFrames] = helperAddNewKeyFrame(mapPointSet, vSetKeyFrames, ...
        currPose, currFeatures, currPoints, mapPointsIdx, featureIdx, localKeyFrameIds);
    
    % Remove outlier map points that are observed in fewer than 3 key frames
    [mapPointSet, directionAndDepth, mapPointsIdx] = helperCullRecentMapPoints(mapPointSet, ...
        directionAndDepth, mapPointsIdx, newPointIdx);
    
    % Create new map points by triangulation
    minNumMatches = 20;
    minParallax   = 3;
    [mapPointSet, vSetKeyFrames, newPointIdx] = helperCreateNewMapPoints(mapPointSet, vSetKeyFrames, ...
        currKeyFrameId, intrinsics, scaleFactor, minNumMatches, minParallax);
    
    
    
end















