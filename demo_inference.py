image_path = './demo/demo.jpg'
sentence = 'the most handsome guy'
weights = './checkpoints/refcoco.pth'
device = 'cuda:0'


from flops_profiler.profiler import get_model_profile

# pre-process the input image
from PIL import Image
import torchvision.transforms as T
import numpy as np
#img = Image.open(image_path).convert("RGB")
img_ndarray = np.random.rand(224,224,3)  # (orig_h, orig_w, 3); for visualization
img = Image.fromarray(np.uint8(img_ndarray*255))
original_w, original_h = 224,224  # PIL .size returns width first and height second

image_transforms = T.Compose(
    [
     T.Resize(480),
     T.ToTensor(),
     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

img = image_transforms(img).unsqueeze(0)  # (1, 3, 480, 480)
img = img.to(device)  # for inference (input)

# pre-process the raw sentence
from bert.tokenization_bert import BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
sentence_tokenized = tokenizer.encode(text=sentence, add_special_tokens=True)
sentence_tokenized = sentence_tokenized[:20]  # if the sentence is longer than 20, then this truncates it to 20 words
# pad the tokenized sentence
padded_sent_toks = [0] * 20
padded_sent_toks[:len(sentence_tokenized)] = sentence_tokenized
# create a sentence token mask: 1 for real words; 0 for padded tokens
attention_mask = [0] * 20
attention_mask[:len(sentence_tokenized)] = [1]*len(sentence_tokenized)
# convert lists to tensors
padded_sent_toks = torch.tensor(padded_sent_toks).unsqueeze(0)  # (1, 20)
attention_mask = torch.tensor(attention_mask).unsqueeze(0)  # (1, 20)
padded_sent_toks = padded_sent_toks.to(device)  # for inference (input)
attention_mask = attention_mask.to(device)  # for inference (input)

# initialize model and load weights
from bert.modeling_bert import BertModel
from lib import segmentation

# construct a mini args class; like from a config file


class args:
    swin_type = 'base'
    window12 = True
    mha = ''
    fusion_drop = 0.0


single_model = segmentation.__dict__['lavt'](pretrained='', args=args)
single_model.to(device)
model_class = BertModel
single_bert_model = model_class.from_pretrained('bert-base-uncased')
single_bert_model.pooler = None

model = single_model.to(device)
bert_model = single_bert_model.to(device)


# inference
import torch.nn.functional as F
last_hidden_states = bert_model(padded_sent_toks, attention_mask=attention_mask)[0]
embedding = last_hidden_states.permute(0, 2, 1)
output = model(img, embedding, l_mask=attention_mask.unsqueeze(-1))


with torch.cuda.device(0):
    batch_size = 1
    flops, macs, params = get_model_profile(model=model, # model
                                    input_shape=(batch_size, 3, 480, 480), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                    args=['l_feats','l_mask'], # list of positional arguments to the model.
                                    kwargs={'l_feats':embedding,'l_mask':attention_mask.unsqueeze(-1)}, # dictionary of keyword arguments to the model.
                                    print_profile=True, # prints the model graph with the measured profile attached to each module
                                    detailed=True, # print the detailed profile
                                    module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                    top_modules=1, # the number of top modules to print aggregated profile
                                    warm_up=10, # the number of warm-ups before measuring the time of each module
                                    as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                    output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                    ignore_modules=None, # the list of modules to ignore in the profiling
                                    func_name='forward') # the function name to profile, "forward" by default, for huggingface generative models, `generate` is used


    print("============================================================================================")
    print(f"FLOPS: {flops}")
    print(f"MACS: {macs}")
    print(f"Params: {params}")
    print("============================================================================================")

output = output.argmax(1, keepdim=True)  # (1, 1, 480, 480)
output = F.interpolate(output.float(), (original_h, original_w))  # 'nearest'; resize to the original image size
output = output.squeeze()  # (orig_h, orig_w)
output = output.cpu().data.numpy()  # (orig_h, orig_w)


# show/save results
def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


output = output.astype(np.uint8)  # (orig_h, orig_w), np.uint8
# Overlay the mask on the image
#visualization = overlay_davis(img_ndarray, output)  # red
#visualization = Image.fromarray(visualization)
# show the visualization
#visualization.show()
# Save the visualization
#visualization.save('./demo/demo_result.jpg')




