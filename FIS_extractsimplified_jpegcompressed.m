clc;
clear all;
close all;
% save start time
start_time=cputime;


file_name='Lena256.jpg';
original_image=imread(file_name);


cover_object1=imread(file_name);
%cover_object1=rgb2gray(cover_object1);
figure;
imshow(cover_object1);title('Original image (before watermark and compression)');

cover_object=im2double(cover_object1);

% determine size of cover image

Mc=size(cover_object,1);	        %Height
Nc=size(cover_object,2);	        %Width

% blocksize=8
blocksize=8; 

% determine maximum message size based on cover object, and blocksize
max_message=Mc*Nc/(blocksize^2);

% process the image in blocks
x=1;
y=1;
sum_dc=0;
  for kk=1:max_message
       
        % transform block 
        image_block=dct2(cover_object(y:y+blocksize-1,x:x+blocksize-1));
        sum_dc=sum_dc + image_block(1,1);
    if (x+blocksize) >= Nc
        x=1;
        y=y+blocksize;
    else
        x=x+blocksize;
    end
        
        end
%claculate mean of all dc cofficients
mean_dc=sum_dc/max_message;

        % process the image in blocks
x=1;
y=1;
sum_dc=0;
  for kk=1:max_message
       
        % transform block 
        image_block=dct2(cover_object(y:y+blocksize-1,x:x+blocksize-1));
        % calculate luminance senstivity
        luminance(1,kk)=(image_block(1,1)/mean_dc)/2;     %------- range(0-2)
        % calculate average contrast
        te=statxture(image_block);
        %average_contrast(1,kk)=te(3);
        % calculate threshold
        threshold(1,kk)=graythresh(image_block);        %-------range(0-1)
        % calculate normalized variance value
        variance(1,kk)=te(3);                            %------- range(0-1)
        % move on to next block. At and of row move to next row
        if (x+blocksize) >= Nc
            x=1;
            y=y+blocksize;
        else
            x=x+blocksize;
        end
        
  end
%FIS system for generating watermark invoked & 
%inputs given to it for each block 
dct_fuz=readfis('watermark');  
for kk=1:max_message
    w(1,kk)=evalfis([ luminance(1,kk) threshold(1,kk) variance(1,kk)],dct_fuz);
end

%Whole 256x256 image's dct taken
dct_cover=dct2(cover_object);
val=dct_cover(1,1);
dct_cover(1,1)=min(min(dct_cover)); %for 2d matrix 1st column wise then row 
watermark=randn(1,max_message);

%sort the DCT of cover image so as to obtain the low frequency components
[svals,idx] = sort(dct_cover(:),'descend'); % sort to vector
lvals=svals;
k=0.05;
%idx returns the corresponding indices of all unsorted elements
%useful in preserving the orignal location of all elements

%Take top kk DCT cofficients and spread the watermark (noise)
for i=1:max_message
    svals(i)=svals(i) + (k * watermark(1,i) * w(1,i));
end



% store position in matrix of top 500000 DCT cofficients
for i=1:max_message
[II,JJ] = ind2sub([Mc,Nc],idx(i)); % position in the matrix
row(i)=II;
col(i)=JJ;
end

% transform the sorted vector again into matrix form
for i=1:(Mc*Nc)
        [II,JJ] = ind2sub([Mc,Nc],idx(i));
        dct_watermark(II,JJ)=svals(i);
end


dct_watermark(1,1)=val;
%Take the inverse DCT
watermarked_image=idct2(dct_watermark);


% convert to uint8 and write the watermarked image out to a file
watermarked_image_int=im2uint8(watermarked_image);
imwrite(watermarked_image_int,'dct_fuzzy.bmp','bmp');

% display processing time
elapsed_time=cputime-start_time,
figure;
imshow(watermarked_image);title('Watermarked image (after watermark added to image)');
%imshow(watermarked_image,[])

i=original_image;
j=imread('dct_fuzzy.bmp');
psnr_var=psnr1(original_image,watermarked_image_int);
ssim_var=ssim(i,j);
disp('PSNR of Original and Watermarked image')
disp(psnr_var)
disp('SSIM of Original and Watermarked image')
disp(ssim_var)


dlmwrite('dct_fuzzywatermark.txt',watermark);
dlmwrite('dct_fuzzyrow.txt',row);
dlmwrite('dct_fuzzycol.txt',col);


I = imread('dct_fuzzy.bmp');
I1=I;
[row coln]= size(I);
I= double(I);
%---------------------------------------------------------
% Subtracting each image pixel value by 128 
%--------------------------------------------------------
I = I - (128*ones(256));

quality = input('What quality of compression you require - ');

%----------------------------------------------------------
% Quality Matrix Formulation
%----------------------------------------------------------
Q50 = [ 16 11 10 16 24 40 51 61;
     12 12 14 19 26 58 60 55;
     14 13 16 24 40 57 69 56;
     14 17 22 29 51 87 80 62; 
     18 22 37 56 68 109 103 77;
     24 35 55 64 81 104 113 92;
     49 64 78 87 103 121 120 101;
     72 92 95 98 112 100 103 99];
 
 if quality > 50
     QX = round(Q50.*(ones(8)*((100-quality)/50)));
     QX = uint8(QX);
 elseif quality < 50
     QX = round(Q50.*(ones(8)*(50/quality)));
     QX = uint8(QX);
 elseif quality == 50
     QX = Q50;
 end
 
%----------------------------------------------------------
% Formulation of forward DCT Matrix and inverse DCT matrix
%----------------------------------------------
DCT_matrix8 = dct(eye(8));
iDCT_matrix8 = DCT_matrix8';   %inv(DCT_matrix8);

%----------------------------------------------------------
% Jpeg Compression
%----------------------------------------------------------
dct_restored = zeros(row,coln);
QX = double(QX);
%----------------------------------------------------------
% Jpeg Encoding
%----------------------------------------------------------
%----------------------------------------------------------
% Forward Discret Cosine Transform
%----------------------------------------------------------

for i1=[1:8:row]
    for i2=[1:8:coln]
        zBLOCK=I(i1:i1+7,i2:i2+7);
        win1=DCT_matrix8*zBLOCK*iDCT_matrix8;
        dct_domain(i1:i1+7,i2:i2+7)=win1;
    end
end
%-----------------------------------------------------------
% Quantization of the DCT coefficients
%-----------------------------------------------------------
for i1=[1:8:row]
    for i2=[1:8:coln]
        win1 = dct_domain(i1:i1+7,i2:i2+7);
        win2=round(win1./QX);
        dct_quantized(i1:i1+7,i2:i2+7)=win2;
    end
end

figure;
imshow(dct_quantized);title('Quantized image while jpeg compression');


%Transmit image 



%-----------------------------------------------------------
% Jpeg Decoding 
%-----------------------------------------------------------
% Dequantization of DCT Coefficients
%-----------------------------------------------------------
for i1=[1:8:row]
    for i2=[1:8:coln]
        win2 = dct_quantized(i1:i1+7,i2:i2+7);
        win3 = win2.*QX;
        dct_dequantized(i1:i1+7,i2:i2+7) = win3;
    end
end
%-----------------------------------------------------------
% Inverse DISCRETE COSINE TRANSFORM
%-----------------------------------------------------------
for i1=[1:8:row]
    for i2=[1:8:coln]
        win3 = dct_dequantized(i1:i1+7,i2:i2+7);
        win4=iDCT_matrix8*win3*DCT_matrix8;
        dct_restored(i1:i1+7,i2:i2+7)=win4;
    end
end
I2=dct_restored;

% ---------------------------------------------------------
% Conversion of Image Matrix to Intensity image
%----------------------------------------------------------

K=mat2gray(I2);

%----------------------------------------------------------
%Display of Results
%----------------------------------------------------------
figure;
imshow(I1);title('Image before jpeg compression')
figure;
imshow(K);title('Restored image after Jpeg decompression');

%-------------------------------------------------------------------------------------
%DECODING THE IMAGE AND COMPARING BOTH PIXEL VALUES by cox method

for i=1:max_message
diff(i)=svals(i)-lvals(i);
end

%Extracted Watermark = (DCT_Low_Frequency_cofficients_Of_Signed_image -
%DCT_Low_Frequency_cofficients_Of_Original image) / (k * weighting factor)

for i=1:max_message
    watermark_n(1,i)=diff(i)/(k*w(1,i));
end
watermark_n(1,1)=0; %dc component is unchanged

SIM=sum(watermark.*watermark_n)/sqrt(sum(watermark.*watermark_n)); 
SIM


decompressed_img=im2uint8(K);
j=I2; %Restored image from jpeg compression : line 231

%psnr1(i,j);
%ssim(i,j);

psnr_var=psnr1(watermarked_image_int,decompressed_img);
ssim_var=ssim(watermarked_image_int,decompressed_img);
disp('PSNR of Watermarked and Decompressed img')
disp(psnr_var)
disp('SSIM of Watermarked and Decompressed img')
disp(ssim_var)

psnr_var=psnr1(original_image,decompressed_img);
ssim_var=ssim(original_image,decompressed_img);
disp('PSNR of Original and Decompressed img')
disp(psnr_var)
disp('SSIM of Original and Decompressed img')
disp(ssim_var)

% display processing time
elapsed_time=cputime-start_time,