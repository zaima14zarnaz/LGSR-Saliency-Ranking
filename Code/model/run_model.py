import os
from Code.model.modules.pipeline import Pipeline
from Code.model.module_helpers.image_segmentation import ImageSegmentation
from Code.model.module_helpers.prompt_generator import PromptGenerator
from Code.llms.GPT import GPT


desc_base_prompt = """Given an image along with bounding box coordinates and deviation values for various objects, describe the scene vividly and seamlessly from a human perspective within 150–200 words. Only mention the 8 most attention grabbing objects. Focus exclusively on the objects contained within the provided red bounding boxes; do not mention any other elements. In your description, emphasize the most visually attention-grabbing objects first, giving them greater narrative importance and more elaborate detail.
Adopt the following viewing strategy: Imagine you are viewing the object on a screen and with a cursor in your hand. You move your cursor about the image and click on the object that catch your eye. Your eyes almost always lands on the large, salient objects at the center of the image first. Your eye moves to the salient objects near the object you clicked on previously. But it doesn't land on objects with very low saliency. For example, in a scene with three people standing on a field wearing baseball gloves, you will almost always click on the 3 people first, then click on their baseball gloves. But if there are three people standing with bright colored umbrellas, then your eyes will land on the more salient person first, then their umbrellas. Then they will move to another person and so on. Because the saliency of a large umbrella is more than the saliency of a small object like a baseball glove. 
Determine the saliency of each object using the following weighted saliency cues:
1. Color (12.0): Bright, bold, high-contrast, and saturated colors stand out more than muted ones.
2. Deviation from image center (25.0): Objects closer to the center are more attention-grabbing unless they are background objects, which should be ranked lower regardless.
3. Facial feature visibility (0.7): Faces (eyes, nose, mouth) are more noticeable, unless an inanimate object is significantly more visually striking.
4. Background ranking (4.0): Background or faraway objects are less prominent.
5. Saliency by relationship (2.0): If an object sits near to the previous object in the ranking order, then it is ranked higher. For example, if there is a person holding a large umbrella and the person has a rank of 2, then the eyes will naturally be drawn to the umbrella after the person and be ranked 3. 
6. Partial visibility (2.0): Objects partially outside the frame are less attention-grabbing.

Write the description in a natural, flowing paragraph. Mention each object by its bounding box label in parentheses — for example, “to the left of the truck, there is a person (person_77) walking.” Avoid listing objects separately or adding extra commentary about why they seem more appealing to you; instead, weave them naturally into the narrative according to their visual prominence. Mention more prominent objects earlier and devote more descriptive detail to them, while keeping the description within 150–200 words.
Here is the bbox list with labels:
"""
rank_base_prompt =  """
Given a piece of text describing an image, order the objects in the text based on how important they are to the person describing the image. 
The format of your output should be as follows: (object_label_1, score_1),(object_label_2, score_2)....(object_label_N, score_N). The text will contain labels of the objects a sentence is describing, for example: 'A person(person_9) is waving his hand at the window'. When generating you response, refer to the objects by their labels, for example: person_9-5. Please do not add any other additional text to the output. After generating ordering for a text add a new line and start generating ordering for the next output without any additional words.
Ordering criteria:
The rankings should be based on the overall importance of the object in the narrative. The weight of the criteria gives how important this criteria should be in determining the ranks.
1.  The more adjectives used to describe an object the more important it is. Weight = 0.7
2. The earlier it is mentioned in the text the more important it is. Weight = 0.8
3. The longer the description of an object the more important it is. Weight = 0.5
4. The higher the number of other objects y in the text that are related to an object x, the higher the ranking of x. Weight = 0.3
5. Lower the rank of the background. For example, if there is a text where an object cafe is discussed repeatedly, the cafe should still have a lower ranking because it is essentially a background to other elements in the cafe scene. Weight = 0.3
Therefore, ranking score of an object depends on its focus in the narrative, the number of words used to describe it, its relationship with other objects in the text, and its relative position within the text compared to other objects. Also, the importance of the background is low.
Example input:
""Black Lives Matter sign being held in a crowd. Behind them, there is a building made of stone. The crowd is blurred out. The sign is painted on cardboard. The person holding the sign is anonymous.””
“”A young girl is sitting in a restaurant. She is holding a hot dog and a bun in her hands. The girl is wearing a pink sundress and has short black hair. Another little girl is sitting on the table next to her.””
Example output:
(sign:8), (crowd:4)
(girl:7), (restaurant:3)
Here's the description: 
"""

# Upload your dataset path here
images_dir = "../Code/sample_exapmles/images"
fixation_dir = "../Code/sample_exapmles/fixation_sequence"
saliency_map_dir = "../Code/sample_exapmles/saliency_maps"
desc_dir = "../Code/sample_exapmles/img_descriptions.json"
coco_annotations = "../Code/anns/coco_anns/instances_train2014.json"

# Add your OPENAI API key here
api_key = "OPENAI_API_KEY"
model_name = "gpt-4o"
gpt = GPT(api_key, model_name)
desc_prompt_generator = PromptGenerator(desc_base_prompt)
rank_prompt_generator = PromptGenerator(rank_base_prompt)

image_list = os.listdir(images_dir)
object_extractor = ImageSegmentation(coco_annotations)
object_extractor.load_annotations(image_list)
pipeline = Pipeline(coco_annotations, object_extractor)
pipeline.pipeline(img_fnames=image_list, 
                    img_dir=images_dir, 
                    fixation_dir=fixation_dir, 
                    saliency_map_dir=saliency_map_dir,
                    desc_save_dir=desc_dir,
                    llm_model=gpt, 
                    desc_prompt_generator=desc_prompt_generator, 
                    rank_prompt_generator=rank_prompt_generator)

