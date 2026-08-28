import gradio as gr
import GenDetectInfer as gdi
import ClipDetectInfer as cdi
import ELAInfer as einf
import NoiseInfer as ninf
import FFTInfer as finf
import MetadataInfer as minf


def run_all_forensics(image, ela_quality=85):
    ela_ov, ela_txt = einf.analyze_ela(image, ela_quality)
    n_ov, n_txt = ninf.analyze_noise(image)
    f_ov, f_txt = finf.analyze_spectrum(image)
    m_txt = minf.analyze_metadata(image)
    return ela_ov, ela_txt, n_ov, n_txt, f_ov, f_txt, m_txt

with gr.Blocks() as interface:

    with gr.Row():
        input_image = gr.Image(type="pil", label="Input ur image here!")
        instructions = gr.Markdown("""
        # READ THIS!
        Hello! I built this to help you detect AI images with ease, however I must remind you to use other methods of verification if needed!!!

        IF YOU WANT TO THESE TOOLS WORK, I HAVE WRITTEN SOME DOCUMENTS ON THE METHODS I USED! THEY ARE VERY INTRESTING :D

        I reccomend "Open"AI's implementation of SynthID (only works for GPT generated images), Google does have their own implementation but as far as im aware you have to use Gemini to use it.
        """)
        with gr.Column():
            synthid = gr.Button("ClosedAI SynthID", link="https://openai.com/research/verify/", link_target="_blank")
            runAll = gr.Button("Run all manual forensic methods")

    with gr.Tab("Vision Transformer (Autodetector)"):
        gr.Markdown("""
        This autodetector uses an AI model called a "Vision Transformer", it CAN potentially be inaccurate. I made the script with the ability to provide grad-cam visuals so you can test and see if
        the things the AI is paying attention to are actually AI. (Red = more attention paid, Blue = less attention paid). Even then, you shouldn't trust grad-cam fully either.
        I personally reccomend using the other tools if you have any doubts at all. You can seriously harm artists by saying their work is AI.""")
        with gr.Row():
            output_grad = gr.Image(label="This is where the model thinks the most AI part of the image is.")
            output_pred = gr.Text(label="What the model thinks the image is.")
            output_prob = gr.Text(label="How confident the model is in its choice. IT CAN BE CONFIDENTLY WRONG, ALWAYS VERIFY THE RESULT!")

        quality = gr.Slider(
            minimum=0, maximum=4, step=1, value=4,
            label="Grad-Cam Accuracy quality",
            info="Higher = more accurate (lower resolution heatmap). Lower = higher-res heatmap (less accurate to what the model viewed during the prediction)."
        )

        useViT = gr.Button("Infer Vision transformer (Auto Detector)")
        useViT.click(
            fn=gdi.detect_ai,
            inputs=[input_image, quality],
            outputs=[output_grad, output_pred, output_prob]
        )

    with gr.Tab("CLIP Based (Autodetector)"):
        gr.Markdown("""
        This autodetector uses an AI model called CLIP. CLIP works by converting images and text into the same mathmatical space. Imagine you had a photo of a car as well as some text
        that says "car", the model basically takes that image and the text and converts them into the same meaning, the text "car" and the image of the car are the same to the model at this point.
        CLIP was never trained to detect AI images, but it turns out its image "fingerprints" separate real and AI images cleanly with a simple rule (check the Methods vault for the full explanation).
        Like the ViT tab, a heatmap shows where the model looked for AI evidence (Red = more, Blue = less). Same disclaimer: it can be confidently wrong, always verify.
        """)
        with gr.Row():
            clip_grad = gr.Image(label="Where CLIP focuses (AI-class saliency).")
            clip_pred = gr.Text(label="What CLIP thinks the image is.")
            clip_prob = gr.Text(label="How confident CLIP is it is AI. IT CAN BE CONFIDENTLY WRONG, ALWAYS VERIFY THE RESULT!")

        useClip = gr.Button("Infer CLIP (Auto Detector)")
        useClip.click(
            fn=cdi.detect_ai_clip,
            inputs=input_image,
            outputs=[clip_grad, clip_pred, clip_prob]
        )

    with gr.Tab("Error Level Analysis (ELA)"):
        gr.Markdown("""
        **No AI model.** Re-saves your image as a JPEG and shows where it compresses *differently* than the rest.
        Real camera JPEGs compress consistently. AI exports / PNGs / edited regions light up bright.
        Not proof on its own — use it as a corroborating clue alongside the detectors. (Full writeup in the Methods vault.)
        """)
        ela_quality = gr.Slider(60, 95, value=85, step=1, label="JPEG re-save quality (lower = more dramatic)")
        ela_img = gr.Image(label="ELA heatmap (bright = compresses differently)")
        ela_txt = gr.Text(label="Analysis")
        ela_btn = gr.Button("Run ELA")
        ela_btn.click(fn=einf.analyze_ela, inputs=[input_image, ela_quality], outputs=[ela_img, ela_txt])

    with gr.Tab("Noise Residual"):
        gr.Markdown("""
        **No AI model.** Estimates the image's noise (image minus a denoised version) and maps its local variance.
        Real photos carry fairly uniform sensor noise; AI images often look *too clean* (low noise) or have noise that
        breaks inconsistently across regions, e.g. edges have some noise, cut off, then again some more noise. Smooth real areas can falsely read as "clean" so please weigh with the other tools. (Methods vault.)
        """)
        noise_img = gr.Image(label="Noise variance map (red = noisy, blue = smooth)")
        noise_txt = gr.Text(label="Analysis")
        noise_btn = gr.Button("Run Noise Analysis")
        noise_btn.click(fn=ninf.analyze_noise, inputs=input_image, outputs=[noise_img, noise_txt])

    with gr.Tab("Frequency Spectrum (FFT)"):
        gr.Markdown("""
        **No AI model.** Shows the image's frequency spectrum (center = large structure, edges = fine detail).
        GAN images can leave grid-like spectral peaks; over-smoothed AI images have too little high-frequency energy.
        Weaker on modern diffusion models, best as one vote among several. (Methods vault.)
        """)
        fft_img = gr.Image(label="Log-magnitude spectrum (centered)")
        fft_txt = gr.Text(label="Analysis")
        fft_btn = gr.Button("Run Spectrum Analysis")
        fft_btn.click(fn=finf.analyze_spectrum, inputs=input_image, outputs=[fft_img, fft_txt])

    with gr.Tab("Metadata / EXIF"):
        gr.Markdown("""
        **No AI model.** Reads the image's EXIF / metadata and flags known AI-tool tags (Midjourney, Stable Diffusion, DALL·E, etc.).
        A present AI tag is strong evidence; absence is weak, most social platforms strip metadata, and AI exports often have none. (Methods vault.)
        """)
        meta_txt = gr.Textbox(label="Metadata report", lines=20)
        meta_btn = gr.Button("Scan Metadata")
        meta_btn.click(fn=minf.analyze_metadata, inputs=input_image, outputs=meta_txt)

    runAll.click(
        fn=run_all_forensics,
        inputs=[input_image, ela_quality],
        outputs=[ela_img, ela_txt, noise_img, noise_txt, fft_img, fft_txt, meta_txt]
    )


interface.launch(share=True) 