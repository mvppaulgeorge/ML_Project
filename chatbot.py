from wit import Wit

access_token = "KJ7I7W7GV4PR37ODQYSZBTVV7MOTJCR3"
client=Wit(access_token)

def backend(intent):
	if intent == 'Project_theme':
	    return "Our Project Theme is Breast Cancer Predictor, it is called Project "
	elif intent == 'breast_cancer':
	    return "Skin cancer is the abnormal growth of skin cells, which most often develops on skin exposed to the sun. But this common form of cancer can also occur on areas of your skin not ordinarily exposed to sunlight."
	elif intent == 'SkinCancer_types': 
	    return "There 7 Major types of Skin Cancer namely- 1. Basal Cell Carcinoma, 2. Squamous Cell Carcinoma, 3. Melanoma, 4. Merkel Cell Carcinoma, 5. Actinic Keratosis, 6.Atypical Fibroxanthoma, 7. Dermatofibrosarcoma Protuberans"
	elif intent == 'SC_bcc':
	    return "Basal cell carcinoma is the most common type of skin cancer and the most frequently occurring of all cancers. Eight out of every 10 skin cancers are basal cell carcinomas, making this form of skin cancer far and away the most common"
	elif intent == 'bcc_sym':
	    return "A basal cell carcinoma will show itself as a change in the skin. It can appear as a pearly white, skin-colored, or pink bump that is somewhat translucent. It can also be a brown, black, or blue lesion with slightly raised borders. On the back or chest, a flat, scaly, reddish patch is more common. BCC basically looks like - 1.) A pearly white, skin-colored, or pink bump on the skin. It will be translucent, meaning you can see through it slightly, and you can often see blood vessels in it. 2.) A brown, black, or blue lesion or a lesion with dark spots. It will have a slightly raised, translucent border. 3.)A flat, scaly, reddish patch of skin with a raised edge. These will occur more commonly on the back or chest. 4.)A white, waxy, scar-like lesion without a clearly defined border. This “morpheaform” basal cell carcinoma is the least common."  
	elif intent == 'bcc_cure':
	    return "Surgery is the typical treatment method. Depending on the size and location of the removed growth, the wound may be sutured closed, covered with a skin graft, or allowed to heal on its own. Medications used for the treatment of basal cell carcinoma (BCC) include antineoplastic agents such as 5-fluorouracil and imiquimod; the photosensitizing agent methyl aminolevulinate cream; and the acetylenic retinoid tazarotene."
	elif intent == 'SC_scc':
	    return "Squamous cell carcinoma is the second most common form of skin cancer. It forms in the squamous cells that make up the middle and outer layer of the skin. Most squamous cell carcinomas result from prolonged exposure to ultraviolet radiation from the sun or tanning beds or lamps. Unlike basal cell carcinomas, squamous cell carcinomas can occur in more wide-ranging locations. " 
	elif intent == 'scc_sym':
	    return "Squamous cell carcinomas appear as red scaly patches, scaly bumps, or open sores. Left alone, they become larger and destroy tissue on the skin. They can also spread to other areas of the body. Other signs of SCC are A firm red nodule, A flat sore with a scaly crust, A new sore or raised area on an old scar, A rough, scaly patch on your lip that can become an open sore, A red sore or rough patch inside your mouth"      
	elif intent == 'scc_cure':
	    return "Although the squamous cell carcinoma needs to be relatively small and superficial, topical treatments can be successful. These drugs work by inflaming the area where they are applied. The body responds by sending white blood cells to attack the inflammation. These white blood cells go after the mutated basal cells. Aldara, Efudex, and Fluoroplex are three of the most used drugs."
	elif intent == 'sc_mcc':
	    return "Also known as neuroendocrine carcinoma of the skin, Merkel cell carcinoma is a rare type of skin cancer. It occurs in the Merkel cells, which are found at the base of the epidermis, the skin’s outermost layer" 
	elif intent == 'mcc_sym':
	    return "Merkel cell carcinoma usually starts on areas of skin exposed to the sun, especially the face, neck, arms, and legs. It first appears as a single pink, red, or purple shiny bump that doesn’t hurt. These can bleed at times. Merkel cell carcinoma is rare, and the first signs of it can look like more common forms of skin cancer that aren’t as aggressive. That makes early detection critical, as in many cases only a biopsy will identify it as Merkel cell carcinoma."
	elif intent == 'mcc_cure':
	    return  "As with melanoma, early diagnosis of Merkel cell carcinoma is imperative to increase the patient’s odds of successful treatment. Excision is the first treatment option for Merkel cell carcinoma. The tumor along with a border of normal skin is removed. This may be done with a standard scalpel excision or it may be done with Mohs surgery to limit the amount of healthy tissue removed and manage future scarring. Over medication, the suggested treatments are chemotherapy and radiation therapy "     
	elif intent == 'SC_melanoma':
	    return "Melanoma is the most dangerous type of skin cancer. It develops in the skin cells that produce melanin, the melanocytes. Exposure to ultraviolet radiation from the sun or from tanning beds increases a person’s risk of developing melanoma. The reason melanoma is more deadly than squamous cell or basal cell carcinoma is that as melanoma progresses it grows downward and can begin to deposit cancerous cells into the bloodstream where they can spread cancer anywhere in the body."
	elif intent == 'melanoma_sym':
	    return "The ABCDE rule is another guide to the usual signs of melanoma. A is for Asymmetry: One half of a mole or birthmark does not match the other, B is for Border: The edges are irregular, ragged, notched, or blurred, C is for Color: The color is not the same all over and may include different shades of brown or black, or sometimes with patches of pink, red, white, or blue, D is for Diameter: The spot is larger than 6 millimeters across (about ¼ inch – the size of a pencil eraser), although melanomas can sometimes be smaller than this, E is for Evolving: The mole is changing in size, shape, or color." 
	elif intent == 'melanoma_cure':
	    return "The treatment of melanoma depends on the size and stage of cancer. If caught early, melanoma can be fully removed during the biopsy. This is especially true if cancer has not started growing downward yet. Again treatments such as Chemotherapy,Immunotherapy and Radiation are preferred over any sort of medication to treat melanoma"
	elif intent == 'SC_Actinic_keratoses':
	    return "Otherwise known as a “precancer,” an actinic keratosis is usually a scaly spot that is found on sun-damaged skin. Actinic keratoses are usually non-tender, may be pink or red and rough, resembling sandpaper. They occur most frequently on the face, scalp, neck, and forearms. Actinic keratoses are considered precursors to squamous cell carcinoma, although most do not progress past the precancer stage"
	elif intent == 'ak_sym':
	    return "Actinic keratosies growths are not painful and are not overly disfiguring because they remain small. These are the signs: Rough, scaly, dry patch of skin, Usually less than 1 inch in diameter, Flat to slightly raised patch or bump atop the skin, Sometimes can be hard and wart-like, Color may be pink, red, or brown, May itch or burn when brushed."    
	elif intent == 'ak_cure':
	    return "Fluorouracil cream, Imiquimod cream, Ingenol mebutate gel, Diclofenac gel are the suggested medication. If the cancer is identified at later stages, surgery is the only effective option left."
	elif intent == 'SC_dp':
	    return "Dermatofibroma sarcoma protuberans is a rare tumor that arises from cells in the dermis and has an unknown cause. It usually presents as a painless, thickened bump in the skin which grows slowly over time. They are red and brown in color and tend to recur after surgical excision unless the margins are definitively cleared. These tumors can have a very extensive deep component and require large margins to clear adequately."
	elif intent == 'dp_sym':
	    return "The causes of this rare form of skin cancer are unknown. There is some thought that dermatofibrosarcoma protuberans can begin on skin that was badly injured from a burn or from surgery. There is not a link between sun exposure and this rare skin cancer"  
	elif intent == "":
	    return "Prior to the development of Mohs methods for excision, there was a high recurrence rate with dermatofibrosarcoma protuberans. That has changed. Even with recurrent dermatofibrosarcoma protuberans, Mohs surgery has a 98 percent cure rate. Medines are not suggested unless a special case"     
    
st.title("Chatbot :pencil2:")  
	st.subheader('Solve your doubts by interacting with our powerful chatbot.')
	if st.checkbox('You can try one of these queries:'):
		st.text('What is your project theme?')
		st.text("What are the types of breast cancer?")
		st.text('Can Invasive Ductal Carcinoma be cured?')
	query = st.text_area('Enter your query','')
	if st.button('Ask Chatbot!'):
		response = client.message(query)
		#st.text(response['intents'])
		#st.text(len(response['intents']))
		if len(response['intents'])!=0:
		
		    intent = response['intents'][0]['name']
		    if intent is None:
		        st.warning('Try framing the question in a different way...')
		    else:
		        output = backend(intent)
		        st.markdown(f"## <font color='blue'>** {output}**</font>",unsafe_allow_html=True)
		else:
		    st.warning('Enter a valid question') 

# Install Transformers

import streamlit as st
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

@st.cache(hash_funcs={transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast: hash}, suppress_st_warning=True)
def load_data():    
 tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
 model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
 return tokenizer, model
tokenizer, model = load_data()

st.write("Welcome to the Chatbot. What is your name?")
input = st.text_input('User:')
if 'count' not in st.session_state or st.session_state.count == 6:
 st.session_state.count = 0 
 st.session_state.chat_history_ids = None
 st.session_state.old_response = ''
else:
 st.session_state.count += 1

new_user_input_ids = tokenizer.encode(input + tokenizer.eos_token, return_tensors='pt')

bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_user_input_ids], dim=-1) if st.session_state.count > 1 else new_user_input_ids

st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

if st.session_state.old_response == response:
   bot_input_ids = new_user_input_ids
 
   st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=5000, pad_token_id=tokenizer.eos_token_id)
   response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
st.write(f"Chatbot: {response}")

st.session_state.old_response = response
