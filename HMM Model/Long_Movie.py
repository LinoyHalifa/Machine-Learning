from moviepy.editor import VideoFileClip, concatenate_videoclips

# Load original video
clip = VideoFileClip("Jump.mp4")

# Repeat 3 times
extended_clip = concatenate_videoclips([clip] * 3)

# Save to new file
extended_clip.write_videofile("jumping.mp4")
