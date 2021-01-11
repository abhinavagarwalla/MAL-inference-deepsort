ffmpeg -framerate 25 -i %d.jpg -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
