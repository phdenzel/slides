#ffmpeg -r 18 -start_number 0 -i cmb3D_1e-4_%d.png -c:v libx265 -crf 40 output.mp4
ffmpeg -framerate 18 -i cmb3D_1e-4_%d.png -pix_fmt yuva420p output.webm
