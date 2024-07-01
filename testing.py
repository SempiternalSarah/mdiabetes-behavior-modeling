import pandas as pd

baseline = pd.read_excel("./arogya_content/preprod_baseline_questionnaires/mDiabetes-baseline-kannada.xlsx")
endline = pd.read_excel("./arogya_content/GAI4SG-Endline-Responses.xlsx")
wapps = pd.read_csv('./arogya_content/all_ai_participants.csv')
combined = pd.merge(baseline, endline, on="18. ನಿಮ್ಮ WhatsApp ಫೋನ್ ಸಂಖ್ಯೆ", suffixes=("_baseline", "_endline"))
wapps["WHATSAPP"] = wapps.apply(lambda x: int(str(x["WHATSAPP"])[2:]), axis=1)
# print(wapps["WHATSAPP"])
combined["18. ನಿಮ್ಮ WhatsApp ಫೋನ್ ಸಂಖ್ಯೆ"] = combined.apply(lambda x: int(x["18. ನಿಮ್ಮ WhatsApp ಫೋನ್ ಸಂಖ್ಯೆ"]), axis=1)
# print(combined["18. ನಿಮ್ಮ WhatsApp ಫೋನ್ ಸಂಖ್ಯೆ"])
combined['AI group'] = combined.apply(lambda x: x["18. ನಿಮ್ಮ WhatsApp ಫೋನ್ ಸಂಖ್ಯೆ"] in wapps["WHATSAPP"].values, axis=1)
print(combined['AI group'].value_counts())
with pd.ExcelWriter('combined.xlsx') as writer:
    combined.to_excel(writer, "combined")