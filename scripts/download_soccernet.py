from SoccerNet.Downloader import SoccerNetDownloader

# Videos will go here
downloader = SoccerNetDownloader(
    LocalDirectory="data/raw_videos"
)

# Option A: Tracking Task (Best for PS1 - Player & Ball movement)
# This usually downloads zip files containing images/videos and labels.
downloader.downloadDataTask(task="tracking", split=["test"])

# Option B: Calibration (If you want pitch keypoints for Homography)
# downloader.downloadDataTask(task="calibration", split=["test"])