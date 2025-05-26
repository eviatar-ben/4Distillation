# Function to get N random image paths from the dataset
base_dir = '/sci/labs/sagieb/eviatar/data/UCF-101-Frames'
def get_random_image_paths(base_dir, N=1000):
    import os
    import random

    # List all the action folders
    action_folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if
                      os.path.isdir(os.path.join(base_dir, folder))]

    # List all subfolders (video folders) in the action folders
    video_folders = []
    for action_folder in action_folders:
        video_folders.extend([os.path.join(action_folder, subfolder) for subfolder in os.listdir(action_folder) if
                              os.path.isdir(os.path.join(action_folder, subfolder))])

    # Now, randomly select N paths
    random_paths = []
    while len(random_paths) < N:
        # Pick a random video folder
        video_folder = random.choice(video_folders)

        # Get all image files in this video folder
        image_files = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith('.jpg')]

        # Pick a random image from this video folder
        if image_files:
            random_image = random.choice(image_files)
            if random_image not in random_paths:  # Ensure uniqueness
                random_paths.append(random_image)

    return random_paths


# Set the number of random images you want to extract
N = 1000  # Change this to the number of random paths you want

# Get the random image paths
random_image_paths = get_random_image_paths(base_dir, N)

# Print the selected paths
for path in random_image_paths:
    print(path)
