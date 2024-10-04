from moviepy.editor import VideoFileClip

def segment_video(input_video_path, output_video_path, start_time, end_time):
    """
    Segments a video into a shorter clip.

    Parameters:
    input_video_path (str): Path to the input video.
    output_video_path (str): Path where the segmented clip will be saved.
    start_time (float): Start time of the segment in seconds.
    end_time (float): End time of the segment in seconds.
    """
    # Load the video file
    clip = VideoFileClip(input_video_path)
    
    # Create the subclip
    subclip = clip.subclip(start_time, end_time)
    
    # Save the subclip to the output path
    subclip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
    
    # Close the clip to release resources
    clip.close()
    subclip.close()


def show_meta(video_path):

    # Replace 'your_video.mp4' with the path to your video file

    # Load the video file
    clip = VideoFileClip(video_path)

    # Extract video metadata
    duration = clip.duration  # duration in seconds
    fps = clip.fps  # frames per second
    size = clip.size  # width and height of the video in pixels
    aspect_ratio = size[0] / size[1]  # width / height
    audio_info = clip.audio is not None  # whether the video has audio or not

    # Print the metadata
    print(f"Duration: {duration} seconds")
    print(f"Frame Rate: {fps} fps")
    print(f"Resolution: {size[0]} x {size[1]}")
    print(f"Aspect Ratio: {aspect_ratio:.2f}")
    print(f"Has Audio: {audio_info}")


if __name__ == '__main__':
    INPUT_VID = "videos/v_2324_223_s2.mp4"
    OUTPUT_VID = "videos/v_2324_223_s2_short.mp4"

    # show meta of SRC_VID
    show_meta(INPUT_VID)

    # segment video
    segment_video(INPUT_VID, OUTPUT_VID, start_time=21, end_time=81)

    # show meta of OUTPUT_VID
    show_meta(OUTPUT_VID)
