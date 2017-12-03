v = VideoReader('sample_v2.mp4');
k=0;
while hasFrame(v)
    frame = readFrame(v);
    myvideo(k)=frame;
    k
    k+1;
end
whos video
 
figure(1);imshow(frame)
I1=frame;
I2 = imresize(I1,[256 256]); 
In=rgb2gray(I2); % use if the image containing RGB value 3
figure(2);imshow(In);
imwrite(In,'yi.jpg') ;