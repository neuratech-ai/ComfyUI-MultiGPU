import torch
import folder_paths
import comfy.sd
import comfy.model_management

current_device = "cuda:0"


def get_torch_device_patched():
    global current_device
    if (
        not torch.cuda.is_available()
        or comfy.model_management.cpu_state == comfy.model_management.CPUState.CPU
    ):
        return torch.device("cpu")

    return torch.device(current_device)


comfy.model_management.get_torch_device = get_torch_device_patched


class CheckpointLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_checkpoint"
    CATEGORY = "loaders"

    def load_checkpoint(self, ckpt_name, device):
        global current_device
        current_device = device

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return out[:3]


class UNETLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("unet"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype, device):
        global current_device
        current_device = device

        dtype = None
        if weight_dtype == "fp8_e4m3fn":
            dtype = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            dtype = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("unet", unet_name)
        model = comfy.sd.load_unet(unet_path, dtype=dtype)
        return (model,)


class VAELoaderMultiGPU:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        return vaes

    @staticmethod
    def load_taesd(name):
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        encoder = next(
            filter(lambda a: a.startswith("{}_encoder.".format(name)), approx_vaes)
        )
        decoder = next(
            filter(lambda a: a.startswith("{}_decoder.".format(name)), approx_vaes)
        )

        enc = comfy.utils.load_torch_file(
            folder_paths.get_full_path("vae_approx", encoder)
        )
        for k in enc:
            sd["taesd_encoder.{}".format(k)] = enc[k]

        dec = comfy.utils.load_torch_file(
            folder_paths.get_full_path("vae_approx", decoder)
        )
        for k in dec:
            sd["taesd_decoder.{}".format(k)] = dec[k]

        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        return sd

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": (s.vae_list(),),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    # TODO: scale factor?
    def load_vae(self, vae_name, device):
        global current_device
        current_device = device

        if vae_name in ["taesd", "taesdxl", "taesd3"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return (vae,)


class ControlNetLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "control_net_name": (folder_paths.get_filename_list("controlnet"),),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "loaders"

    def load_controlnet(self, control_net_name, device):
        global current_device
        current_device = device

        controlnet_path = folder_paths.get_full_path("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        return (controlnet,)


class CLIPLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "type": (
                    ["stable_diffusion", "stable_cascade", "sd3", "stable_audio"],
                ),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, device, type="stable_diffusion"):
        global current_device
        current_device = device

        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        clip_path = folder_paths.get_full_path("clip", clip_name)
        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        return (clip,)


class DualCLIPLoaderMultiGPU:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("clip"),),
                "clip_name2": (folder_paths.get_filename_list("clip"),),
                "type": (["sdxl", "sd3", "flux"],),
                "device": ([f"cuda:{i}" for i in range(torch.cuda.device_count())],),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name1, clip_name2, type, device):
        global current_device
        current_device = device

        clip_path1 = folder_paths.get_full_path("clip", clip_name1)
        clip_path2 = folder_paths.get_full_path("clip", clip_name2)
        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
        )
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "CheckpointLoaderMultiGPU": CheckpointLoaderMultiGPU,
    "UNETLoaderMultiGPU": UNETLoaderMultiGPU,
    "VAELoaderMultiGPU": VAELoaderMultiGPU,
    "ControlNetLoaderMultiGPU": ControlNetLoaderMultiGPU,
    "CLIPLoaderMultiGPU": CLIPLoaderMultiGPU,
    "DualCLIPLoaderMultiGPU": DualCLIPLoaderMultiGPU,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CheckpointLoaderMultiGPU": "Load Checkpoint (Multi-GPU)",
    "UNETLoaderMultiGPU": "Load Diffusion Model (Multi-GPU)",
    "VAELoaderMultiGPU": "Load VAE (Multi-GPU)",
    "ControlNetLoaderMultiGPU": "Load ControlNet Model (Multi-GPU)",
    "CLIPLoaderMultiGPU": "Load CLIP (Multi-GPU)",
    "DualCLIPLoaderMultiGPU": "DualCLIPLoader (Multi-GPU)",
}
