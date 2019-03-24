% Practical 1: Harris Corner Detection
% Change the name of the image in imread() and in image_name.
% To save images to the path, unquote respective saveas() lines.

% ------------------------- Part 1 ---------------------------%
I = imread('chessboard00.png'); % Read image 

image_name = 'chessboard00'; % image name (I defined name seperately so that I can save 
% plots by concatenating image name automatically e.g. strcat(image_name,'corners.png') will 
% output file chessboard00corners.png

Im = imread('chessboard00.png'); % Read image again because we are converting some rgb images to
% grayscale, and we need original image to show salient points on those
% images. Done to not change code everytime.

if size(I,3)>1 % Images 5 and 6 are in rgb so we need to convert them  
    % to grayscale else conv2 gives error that N-D arrays are not supported.
    I=rgb2gray(I); 
end

% Compute the derivatives

% In direction of x
dx = [-1 0 1; -1 0 1; -1 0 1]; 

% In direction of y
dy = [-1 -1, -1; 0 0 0; 1 1 1];

% Filter image with these derivatives i.e. Calculate image derivatives 

Ix = conv2(double(I), dx, 'same'); % Derivative in x-direction
Iy = conv2(double(I), dy, 'same'); % Derivative in y-direction

% Generate Gaussian filter of size 9*9 and standard deviation sigma = 2
sigma = 2;
g = fspecial('gaussian',9, sigma); % Gaussian filter

% Smooth second order image derivatives Ix*Ix, Iy*Iy and Ix*Iy

Ix2 = conv2(Ix.*Ix, g, 'same'); % Returns the two dimensional convolution that is the same size as Ix.^2.
Iy2 = conv2(Iy.*Iy, g, 'same'); 
Ixy = conv2(Ix.*Iy, g, 'same');

% I'll use subplot to plot original + derivatives in same plot.

FigHandle = figure;
set(FigHandle, 'Position', [100, 100, 1049, 600],'Name',['Corner Detection ']);
subplot(2,3,1);imshow(mat2gray(I),[]);title(image_name);
hold on 
subplot(2,3,2); imshow(mat2gray(Ix2),[]);title('Ix');
hold on
subplot(2,3,3); imshow(mat2gray(Ix2),[]);title('Ix');
hold on
subplot(2,3,4); imshow(mat2gray(Ix2),[]);title('Ix*Ix - Gaussian filter');
hold on
subplot(2,3,5); imshow(mat2gray(Iy2),[]);title({'Squared derivative Iy*Iy'; 'Gaussian filter sigma=2'});
hold on
subplot(2,3,6); imshow(mat2gray(Ixy),[]), title({'Squared derivative Ix*Iy'; 'Gaussian filter sigma=2'});
hold on

% saveas(FigHandle, strcat(image_name, 'derivatives.png'), 'png') % Saves output image
% To save plot, unquote above line



% ------------------- Part 2 ------------------ %

%Compute E, the matrix that contains for each pixel the value of the smaller eigenvalue of M. Display the
%matrix E.
[i_height, i_width] = size(I);
E = zeros( i_height, i_width); % Initialize E
filter = 1/9*[1 1 1 ;1 1 1 ;1 1 1]; % Define filter to calculate E
Ix2_c = conv2(Ix2, filter, 'same'); 
Iy2_c = conv2(Iy2, filter, 'same'); 
Ixy_c = conv2(Ixy, filter, 'same');


tic % Time start 
 for i = 1:i_height
    for j = 1:i_width
        M = [Ix2_c(i,j) Ixy_c(i,j); Ixy_c(i,j) Iy2_c(i,j)];
        eig_v = min(eig(M));
        E(i,j) = eig_v;
    end
 end

elapsed_time_E = toc; % Time end

FigHandle2 = figure;
set(FigHandle2, 'Position', [100, 100, 1049, 600],'Name',['Corner Detection ']);
subplot(2,4,1);imshow(mat2gray(I),[]);title(image_name)
hold on
subplot(2,4,2);imshow(mat2gray(E),[]);title(['E (Elapsed Time= ',num2str(elapsed_time_E),'s)']); % Plot on the same previous plot
hold on




% ------------------------- Part 3 ---------------------------

% Compute Matrix R which contains for every point the cornerness score

k = 0.04; % k is a constant

tic % Time start
R = zeros(i_height, i_width); % initialize R 
% R will be calulcated in loop and then using this, we'll 
for i = 1:i_height
    for j = 1:i_width
        M = [Ix2_c(i,j), Ixy_c(i,j); Ixy_c(i,j), Iy2_c(i,j)];
        R(i,j) = double(det(M) - k*trace(M)^2);
    end
end
elapsed_time_R = toc;
subplot(2,4,3);imshow(mat2gray(R),[]);title(['R (Elapsed-Time= ',num2str(elapsed_time_R),' s)']);
hold on
% Computation of R can be made faster by calculating without loop
tic % Time start
ROptimzed = (Ix2_c.* Iy2_c+ Ixy_c.*Ixy_c) - k*(Ix2_c +Iy2_c).^2;
elapsed_time_R_optimized = toc;
subplot(2,4,4);imshow(mat2gray(R),[]);title(['R (Optimized-Time= ',num2str(elapsed_time_R_optimized),' s)']); 
hold on

% ------------------------- Part 4 ------------------------------------
% For E and R, select the 81 most salient points. Do you get the result that you expected?
% Maximal supression

points = 81; % Number of points to be selected
tic
features_E_Max = struct('p_x', zeros(points,1), 'p_y', zeros(points, 1)); 
[~, index_E] = sort( E(:), 'descend');
[row_E,col_E] = ind2sub(size(E), index_E);
for i = 1: points  
    features_E_Max(i).p_y = row_E(i);   
    features_E_Max(i).p_x = col_E(i);  
end
toc
subplot(2,4,5);imshow(Im,[]);title({'E - 81 most salient points'}); 
hold on;
for i = 1: size(features_E_Max, 2)    
    plot(features_E_Max(i).p_x, features_E_Max(i).p_y, 'g+'); 
end

tic
features_R_Max = struct('p_x', zeros(points, 1), 'p_y', zeros(points, 1));
[~, index_R] = sort( R(:), 'descend');
[row_R,col_R] = ind2sub(size(R), index_R);
for i = 1: points  
    features_R_Max(i).p_y = row_R(i);   
    features_R_Max(i).p_x = col_R(i);  
end
toc

subplot(2,4,6);imshow(Im,[]);title({'R - 81 most salient points'}); 
hold on;
for i = 1: size(features_R_Max, 2)    
    plot(features_R_Max(i).p_x, features_R_Max(i).p_y, 'g+'); 
end
title('R - 81 most salient points')

% Part 4- Non-maximal

window = 11; % Select a window of size 11 by 11
features_E_NonM = struct('p_x', zeros(points, 1), 'p_y', zeros(points, 1)); % Non-maximal 
wsize = floor(window/2);
E_pad = padarray(E,[wsize,wsize]);

count = 1;

while(count<points)
    for i= 1:size(row_E)  
        if( E_pad(row_E(i),col_E(i)) ~= 0 )
            E_pad(row_E(i)-wsize:row_E(i)+wsize, col_E(i)-wsize:col_E(i)+wsize) = 0;
            features_E_NonM(count).p_y = row_E(i);   
            features_E_NonM(count).p_x = col_E(i);
            count = count + 1;       
        end
        if count == 82
            break;
        end
    end
end

subplot(2,4,7);imshow(Im,[]);title({'Non Max - E'; '81 Most salient points'});
hold on;
for i = 1: size(features_E_NonM, 2)    
   plot(features_E_NonM(i).p_x, features_E_NonM(i).p_y, 'g+'); 
end

features_R_NonM = struct('p_x', zeros(81, 1), 'p_y', zeros(points, 1));
R_pad = padarray(R,[wsize,wsize]);
count = 1;
 while(count<points)
    for i= 1:size(row_R)  
        if( R_pad(row_R(i),col_R(i)) ~= 0 )
            R_pad(row_R(i)-wsize:row_R(i)+wsize, col_R(i)-wsize:col_R(i)+wsize) = 0;
            features_R_NonM(count).p_y = row_R(i);   
            features_R_NonM(count).p_x = col_R(i);
            count = count + 1;       
        end
        if count == 82
            break;
        end
    end
 end
subplot(2,4,8);imshow(Im,[]);title({'Non Max - R'; '81 Most salient points'});
hold on;
for i = 1: size(features_R_NonM, 2)    
    plot(features_R_NonM(i).p_x, features_R_NonM(i).p_y, 'g+'); 
end

%saveas(FigHandle2, strcat(image_name, 'corners.png'), 'png') % Saves output image
%To save plot unquote above line