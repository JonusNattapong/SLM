import gradio as gr
import torch
import os
from inference import ThaiSLMInference


class GradioInterface:
    """Gradio web interface for Thai SLM"""
    
    def __init__(self, model_path: str = "./thai_slm_moe_model"):
        try:
            self.inference = ThaiSLMInference(model_path)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
    
    def generate_text(self, 
                     prompt: str,
                     max_length: int = 150,
                     temperature: float = 0.8,
                     top_k: int = 50,
                     top_p: float = 0.9) -> str:
        """Generate text with parameters"""
        
        if not self.model_loaded:
            return "❌ Model not loaded. Please train the model first."
        
        if not prompt.strip():
            return "⚠️ Please enter a prompt."
        
        try:
            generated = self.inference.generate_text(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True
            )
            return generated
        except Exception as e:
            return f"❌ Error generating text: {str(e)}"
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Noto Sans Thai', sans-serif;
        }
        .title {
            text-align: center;
            color: #2E86AB;
            margin-bottom: 20px;
        }
        """
        
        with gr.Blocks(css=css, title="Thai SLM MoE") as interface:
            
            gr.Markdown(
                """
                # 🇹🇭 Thai Small Language Model with Mixture of Experts
                
                ภาษาไทย Small Language Model (SLM) พร้อม Mixture of Experts (MoE) Architecture
                
                ใส่ข้อความเพื่อให้โมเดลต่อเติมหรือตอบคำถาม
                """,
                elem_classes=["title"]
            )
            
            if not self.model_loaded:
                gr.Markdown(
                    """
                    ⚠️ **Model not found!**
                    
                    Please train the model first by running:
                    ```bash
                    python train.py
                    ```
                    """,
                    elem_classes=["warning"]
                )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    prompt_input = gr.Textbox(
                        label="📝 ข้อความเริ่มต้น (Prompt)",
                        placeholder="เช่น: ประเทศไทยมีจังหวัด...",
                        lines=3,
                        max_lines=5
                    )
                    
                    # Parameters section
                    with gr.Accordion("⚙️ พารามิเตอร์การสร้างข้อความ", open=False):
                        max_length = gr.Slider(
                            minimum=10,
                            maximum=500,
                            value=150,
                            step=10,
                            label="ความยาวสูงสุด (Max Length)"
                        )
                        
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="อุณหภูมิ (Temperature)"
                        )
                        
                        top_k = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="Top-K"
                        )
                        
                        top_p = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top-P"
                        )
                    
                    generate_btn = gr.Button(
                        "🚀 สร้างข้อความ",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Output section
                    output_text = gr.Textbox(
                        label="📄 ข้อความที่สร้างขึ้น",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
            
            # Examples section
            with gr.Accordion("💡 ตัวอย่างการใช้งาน", open=True):
                examples = [
                    ["ประเทศไทยมีจังหวัด"],
                    ["วิทยาศาสตร์และเทคโนโลยี"],
                    ["การศึกษาในยุคดิจิทัล"],
                    ["อาหารไทยที่มีชื่อเสียง"],
                    ["ประวัติศาสตร์ของไทย"],
                    ["ศิลปะและวัฒนธรรมไทย"],
                    ["เศรษฐกิจไทยในอนาคต"]
                ]
                
                gr.Examples(
                    examples=examples,
                    inputs=[prompt_input],
                    label="คลิกเพื่อใช้ตัวอย่าง"
                )
            
            # Model info section
            with gr.Accordion("ℹ️ ข้อมูลโมเดล", open=False):
                gr.Markdown(
                    """
                    ### Architecture Details:
                    - **Model Type**: Small Language Model with Mixture of Experts
                    - **Language**: Thai (ไทย)
                    - **Training Data**: ZombitX64/Wikipedia-Thai
                    - **Tokenizer**: Custom ByteLevelBPE for Thai
                    - **Features**: RoPE, SwiGLU, MoE layers
                    
                    ### Parameters:
                    - **Temperature**: ควบคุมความสร้างสรรค์ (สูง = สร้างสรรค์มากขึ้น)
                    - **Top-K**: จำนวนคำที่พิจารณา (น้อย = เลือกเฉพาะคำที่น่าจะเป็นไปได้มาก)
                    - **Top-P**: สัดส่วนความน่าจะเป็นสะสม (น้อย = เลือกคำที่แน่นอนมากขึ้น)
                    """
                )
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_text,
                inputs=[prompt_input, max_length, temperature, top_k, top_p],
                outputs=[output_text]
            )
            
            # Auto-generate on example click
            for example in examples:
                prompt_input.change(
                    fn=lambda x: x if x else "",
                    inputs=[prompt_input],
                    outputs=[]
                )
        
        return interface
    
    def launch(self, share: bool = False, port: int = 7860):
        """Launch the interface"""
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_port=port,
            server_name="0.0.0.0" if share else "127.0.0.1"
        )


def main():
    """Main function to launch the web interface"""
    
    # Check if model exists
    model_path = "./thai_slm_moe_model"
    
    if not os.path.exists(model_path):
        print("⚠️ Model not found!")
        print("Please train the model first by running: python train.py")
        print("Launching interface anyway for demonstration...")
    
    # Create and launch interface
    app = GradioInterface(model_path)
    app.launch(
        share=False,  # Set to True to create public link
        port=7860
    )


if __name__ == "__main__":
    main()
