%implay(mov)

v = VideoWriter('newfile.avi');
open(v)
writeVideo(v,K_array)
close(v)
