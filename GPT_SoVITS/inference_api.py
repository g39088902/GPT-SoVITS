# -*- coding:utf-8 -*-
import os
import re

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import nanoid
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
import soundfile as sf
from tools.i18n.i18n import I18nAuto

app = Flask(__name__)
CORS(app)
change_gpt_weights(gpt_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt")
change_sovits_weights(sovits_path="GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")
i18n = I18nAuto()
lock = ""


def replace_decimal_point(text):
    # 定义正则表达式，用于匹配数字.数字的模式
    pattern = r'\d+\.\d+'
    # 使用正则表达式查找所有匹配的数字
    matches = re.findall(pattern, text)
    # 遍历所有匹配的数字，并替换小数点
    for match in matches:
        replaced_match = match.replace('.', '点')
        text = text.replace(match, replaced_match)
    return text


@app.route('/tts', methods=['POST'])
async def tts():
    target_text = request.json.get('message')
    target_text = replace_decimal_point(target_text)
    target_text = target_text.replace("—", "")
    target_text = target_text.replace("......", "...")
    target_text = target_text.replace("...", "。")
    target_text = target_text.replace("……", "…")
    target_text = target_text.replace("…", "。")
    target_text = target_text.replace("　", "")
    target_text = target_text.replace("、", ",")
    target_text = re.sub(r'[a-zA-Z「」（）“”*/\\\[\]]', '', target_text)
    file_name_text = target_text[:20]
    if os.path.exists(f"C:/Users/Lima/PycharmProjects/GPT-SoVITS/output/output_{file_name_text}.wav"):
        print("命中缓存:",file_name_text)
        return send_file(f"C:/Users/Lima/PycharmProjects/GPT-SoVITS/output/output_{file_name_text}.wav",
                         as_attachment=True)

    global lock
    if lock != "":
        print("busy")
        return jsonify({"message": "busy"})
    else:
        lock = nanoid.generate()

    # Synthesize audio
    synthesis_result = get_tts_wav(
        ref_wav_path="C:/Users/Lima/Downloads/绝区零语音包1.0（中）/妮可/ffecc22bf76bda56.wav",
        prompt_text="园景实业一直想要依靠工程业绩提升拓普死排名，可为了财联的椅子，怎么连最基本的底线都没有了。",
        prompt_language=i18n("中文"),
        text=target_text,
        text_language=i18n("中文"),
        how_to_cut=i18n("按标点符号切")
    )
    lock = ""
    result_list = list(synthesis_result)

    if result_list:
        last_sampling_rate, last_audio_data = result_list[-1]
        output_wav_path = os.path.join("output/", f"output_{file_name_text}.wav")
        sf.write(output_wav_path, last_audio_data, last_sampling_rate)
        print(f"Audio saved to {output_wav_path}")
    return send_file(f"C:/Users/Lima/PycharmProjects/GPT-SoVITS/output/output_{file_name_text}.wav", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=35629)
