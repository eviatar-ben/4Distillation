# TIME_TOKEN = "<time_"
TIME_TOKEN = "<frame_"
PRETRAINED_MODE0_OBJ = False
PROJECTION_DIMENSION = 100

GENERATED_VIDEOS_PATHS = ['/sci/labs/sagieb/eviatar/data/dnerf/time/jumpingjacks/00029.png',
                          '/sci/labs/sagieb/eviatar/data/UCF-101-Frames/PullUps/v_PullUps_g04_c02/frame_00008.jpg',
                          '/sci/labs/sagieb/eviatar/data/UCF-101-Frames/PullUps/v_PullUps_g17_c04/frame_00019.jpg',
                          '/sci/labs/sagieb/eviatar/data/UCF-101-Frames/PullUps/v_PullUps_g16_c03/frame_00027.jpg',
                          '/sci/labs/sagieb/eviatar/data/UCF-101-Frames/Punch/v_Punch_g03_c02/frame_00019.jpg']

UNET_LAYERS = ['IN01', 'IN02', 'IN04', 'IN05', 'IN07', 'IN08', 'MID', 'OUT03', 'OUT04',
               'OUT05', 'OUT06', 'OUT07', 'OUT08', 'OUT09', 'OUT10', 'OUT11']

SD_INFERENCE_TIMESTEPS = [999, 979, 959, 939, 919, 899, 879, 859, 839, 819, 799, 779, 759, 739, 719,
                          699, 679, 659, 639, 619, 599, 579, 559, 539, 519, 500, 480, 460, 440, 420,
                          400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200, 180, 160, 140, 120,
                          100, 80, 60, 40, 20]
TIME_TRAIN_IDX = [0, 15, 20, 45, 60, 70, 80, 85, 100]
TIME_TEST_IDX = [10, 30, 50]
TIME_SPLIT_IDXS = {'test': TIME_TEST_IDX, 'train': TIME_TRAIN_IDX}

# DTU test set scans / scenes. For the train set, we have a list of 'exclusion'
# scenes that are similar to the test scenes (according to PixelNerf authors) and
# the remaining images are train set.
TEST_SET_SCANS = [8, 21, 30, 31, 34, 38, 40, 41, 45, 55, 63, 82, 103, 110, 114]
TRAIN_SET_EXCLUDE_SCANS = [1, 2, 7, 25, 26, 27, 29, 39, 51, 54, 56, 57, 58, 73, 83, 111, 112, 113, 115, 116, 117]
# see the pixelNerf supplementary
#
DTU_MASKS = "data/dtu/submission_data/idrmasks"

PROMPTS = ["A photo of a {}",
           "A photo of {} in the jungle",
           "A photo of {} on a beach",
           "A photo of {} in Times Square",
           "A photo of {} in the moon",
           "A painting of {} in the style of Monet",
           "Oil painting of {}",
           "A Marc Chagall painting of {}",
           "A manga drawing of {}",
           'A watercolor painting of {}',
           "A statue of {}",
           "App icon of {}",
           "A sand sculpture of {}",
           "Colorful graffiti of {}",
           "A photograph of two {} on a table", ]

VALIDATION_PROMPTS = ["A photo of a {}",
                      "A photo of a {} on a beach",
                      "App icon of {}",
                      "A painting of {} in the style of Monet", ]

IMAGENET_TEMPLATES_SMALL = ["a photo of a {}",
                            "a rendering of a {}",
                            "a cropped photo of the {}",
                            "the photo of a {}",
                            "a photo of a clean {}",
                            "a photo of a dirty {}",
                            "a dark photo of the {}",
                            "a photo of my {}",
                            "a photo of the cool {}",
                            "a close-up photo of a {}",
                            "a bright photo of the {}",
                            "a cropped photo of a {}",
                            "a photo of the {}",
                            "a good photo of the {}",
                            "a photo of one {}",
                            "a close-up photo of the {}",
                            "a rendition of the {}",
                            "a photo of the clean {}",
                            "a rendition of a {}",
                            "a photo of a nice {}",
                            "a good photo of a {}",
                            "a photo of the nice {}",
                            "a photo of the small {}",
                            "a photo of the weird {}",
                            "a photo of the large {}",
                            "a photo of a cool {}",
                            "a photo of a small {}", ]

IMAGENET_STYLE_TEMPLATES_SMALL = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]
