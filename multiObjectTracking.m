function multiObjectTracking()
% Initialization functions
clc
obj = setupSystemObjects();
trk = initializeTracks();
db = database();

% Variables
ntrk = 1; 
A = 0; % Car Width Matrix
B = 0; % Car Height Matrix
C = 0; % Car Comparison Table
D = 0;
itrtn = 0;

% Setting strings > 0
carwidth = '...'; 
carheight = '...';
cartype = '...';

% Main loop
while ~isDone(obj.src)
        frame = readFrame();
        [centroids, bboxes, mask] = detectObjects(frame);
        predictNewLocationsOfTracks();
        [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();
        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();
        displayTrackingResults();
end

% Load train car database from file 'compare_cars.dat', comma delimiter
function db = database()
        fileID = 'compare_cars.dat';
        delimiterIn = ',';
        headerlinesIn = 1;
        db = importdata(fileID, delimiterIn, headerlinesIn);
end

% Setup OBJ to contain reader, players, foreground detector, and blob analyzer
% Reading in file '3.mp4'
function obj = setupSystemObjects()
        obj.src = vision.VideoFileReader('3.mp4');
        obj.videoPlayer = vision.VideoPlayer('Position', [500, 200, 700, 600],'Name', 'TrainTracking');
        obj.maskPlayer = vision.VideoPlayer('Position', [740, 400, 700, 400]);
        obj.detect1 = vision.ForegroundDetector('NumGaussians', 2, 'NumTrainingFrames', 3, 'MinimumBackgroundRatio', 0.3, 'AdaptLearningRate', true, 'LearningRate', 0.000001); 
        obj.blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, 'AreaOutputPort', true, 'CentroidOutputPort', true, 'MinimumBlobArea', 10000);
        

end

% Create empty arrays for what I call 'Tracks', which are just arrays of
% variables from each detected box
function tracks = initializeTracks()
        tracks = struct('id', {}, 'bbox', {}, 'kalFilter', {}, 'age', {}, 'totalVisibleCount', {}, 'NoObject', {});
end

% Read frame by frame
function frame = readFrame()
        frame = obj.src.step();
end

% General masking, bbox positioning and noise reduction
function [centroids, bboxes, mask] = detectObjects(frame)
        % Detect foreground.
        mask = obj.detect1.step(frame);
        % Apply mask operations to remove noise and fill in holes/gaps.
        mask = imopen(mask, strel('rectangle', [6,6]));
        mask = imclose(mask, strel('rectangle', [14, 14]));
        mask = imfill(mask, 'holes');
        % Perform blob analysis to find connected components.
        [~, centroids, bboxes] = obj.blobAnalyser.step(mask);
end

% Kalman filter to guestimate new location of box using vectors
function predictNewLocationsOfTracks()
        for i = 1:length(trk)
            bbox = trk(i).bbox;
            % Kalman filter for box
            predictedCentroid = predict(trk(i).kalFilter);
            % Move the bounding box to the predicted spot
            predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
            trk(i).bbox = [predictedCentroid, bbox(3:4)];
        end
end

% Further Kalman filter best-guess analysis
function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment()
        nTracks = length(trk);
        nDetections = size(centroids, 1);
        % Compute the cost of assigning each detection to each track in
        % terms of its position change (drastic = higher cost)
        cost = zeros(nTracks, nDetections);
        for i = 1:nTracks
            cost(i, :) = distance(trk(i).kalFilter, centroids);
        end
        % Actually solve the assignment problem.
        costOfNonAssignment = 20;
        [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);
end

% Increase age and visibility count, and update track variables
function updateAssignedTracks()
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trkIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            correct(trk(trkIdx).kalFilter, centroid);
            trk(trkIdx).bbox = bbox;
            % Add 1 to track age            
            trk(trkIdx).age = trk(trkIdx).age + 1;
            % Add one to currently visible object(s) count
            trk(trkIdx).totalVisibleCount = trk(trkIdx).totalVisibleCount + 1;
            % Zero the no visible objects frame count
            trk(trkIdx).NoObject = 0; 
        end
end

% If no object, increase no object count and age count
function updateUnassignedTracks()
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            trk(ind).age = trk(ind).age + 1;
            trk(ind).NoObject = trk(ind).NoObject + 1;
        end
end

% If object goes out of frame for too long, clear it
function deleteLostTracks()
        if isempty(trk)
            return;
        end
        invisible = 10; % Invisible for too long
        ageThresh = 8;
        % Compute the fraction of the track's age for which it was visible.
        ages = [trk(:).age];
        totalVisibleCounts = [trk(:).totalVisibleCount];
        visibility = totalVisibleCounts ./ ages;
        % Find the index of 'lost' tracks.
        lostInds = (ages < ageThresh & visibility < 0.6) | [trk(:).NoObject] >= invisible;
        % Delete lost tracks.
        trk = trk(~lostInds);
end
  
% Brand new track initialization
function createNewTracks()
        centroids = centroids(unassignedDetections, :);
        bboxes = bboxes(unassignedDetections, :);
        for i = 1:size(centroids, 1)
            centroid = centroids(i,:);
            bbox = bboxes(i, :);
            % Create a Kalman filter object.
            kalFilter = configureKalmanFilter('ConstantVelocity', centroid, [200, 20], [100, 25], 20);
            % Create a new track, age 1, no object found 0
            newTrack = struct('id', ntrk, 'bbox', bbox, 'kalFilter', kalFilter, 'age', 1, 'totalVisibleCount', 1, 'NoObject', 0);
            % Add it to the array of tracks. (Yes I know it is iterative,
            % thank you Matlab, but unless I calculate out the number of 
            % frames with a car in it by hand, we're going to stick with this)
            trk(end + 1) = newTrack;
            ntrk = ntrk + 1;
        end
end
   
% Create frame, display bboxes, and all other info. 
function displayTrackingResults()
        % Convert the frame and the mask to uint8 RGB.
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        minVisibleCount = 15; % reliabile track age
        if ~isempty(trk)
            % Only display tracks that have been visible for more than
            % a minimum number of frames.
            reliableTrackInds = [trk(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = trk(reliableTrackInds);
            % Display the objects. If an object has not been detected
            % in this frame, display its predicted bounding box.
            if ~isempty(reliableTracks)
                % Get bounding boxes.
                
                bboxes = cat(1, reliableTracks.bbox);
                % Get Tracking IDs.
                ids = int32([reliableTracks(:).id]);
                % Create labels for objects indicating the ones for
                % which we display the predicted rather than the actual
                % location.
                labels = cellstr(int2str(ids'));
                predictedTrackInds = [reliableTracks(:).NoObject] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' unreliable'};
                labels = strcat(labels, isPredicted, '. Reliable Car');
                carcount = cellstr(int2str(ids(end)));
                cnttxt = ['Total count of reliable cars: ', carcount];
                carsize = cellstr(int2str(ids(end)));            
                sizetxt = ['Size of #', carsize, 'Width: ', carwidth, 'Height: ', carheight];
               
                % Draw the objects on the frame
                if strcmp(isPredicted,' unreliable')
                    frame = insertObjectAnnotation(frame, 'rectangle',  bboxes, labels, 'color', 'yellow');
                end
                
                if ~strcmp(isPredicted,' unreliable')
                    frame = insertObjectAnnotation(frame, 'rectangle',  bboxes, labels, 'color', 'green');
                end
                
                % Size calculation, now working
                if itrtn == 8 
                    props = regionprops(frame,'BoundingBox');
                    cwidth = props(end).BoundingBox(:,4);
                    cheight = props(end).BoundingBox(:,5);
                    A = cat(1, A, cwidth); % Car width area used for averaging
                    B = cat(1, B, cheight); % Same for height
                    cwidth = mean(A);
                    cheight = mean(B);
                    whratio = cwidth./cheight;
                    % carwidth = carwidthraw .* distancecalculationstuff;
                    % carlength = carlengthraw .* distancecalculationstuff;
                    carwidth = cellstr(int2str(cwidth));
                    carheight = cellstr(int2str(cheight));
                    itrtn = 0; 
                    sizeDB = numel(db.data(:));
                    for i = 1:1:sizeDB
                        D = (whratio ./ db.data(i));
                        C = cat(1, C, D);                                                  
                    end                    
                    tmp = abs(C - 1);
                    [~, idx] = min(tmp(:));
                    idz = idx - 1;
                    A = 0;
                    B = 0;
                    C = 0;  
                    cartype = cellstr(db.colheaders(1,idz));
                end
                
                cartype2 = ['Type of Car: ', cartype];
                itrtn = itrtn+1;
                frame = insertText(frame, [3, 5; 176, 5], cnttxt, 'TextColor', 'white', 'BoxColor', 'black');
                frame = insertText(frame, [3, 26; 61, 26; 82, 26; 128, 26; 165, 26; 219, 26], sizetxt, 'TextColor', 'white', 'BoxColor', 'black');
                frame = insertText(frame, [3, 47; 95, 47], cartype2, 'TextColor', 'white', 'BoxColor', 'black');
                
                % Draw the objects on the mask.
                % mask = insertObjectAnnotation(mask, 'rectangle', bboxes, labels); 
            end
        end
        % Display the mask and/or the frame.
        % obj.maskPlayer.step(mask);
        obj.videoPlayer.step(frame);
end
end

    