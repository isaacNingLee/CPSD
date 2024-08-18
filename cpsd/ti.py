from diffusers import StableDiffusionPipeline

class TIPipeline(StableDiffusionPipeline):

    def load_ti_embed(self, ti_embed_path: str):
        
        self.load_textual_inversion(ti_embed_path)

