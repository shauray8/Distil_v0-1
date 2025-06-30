## replace denoise function on orig FLUX repo for FlowDPM

def denoise(
    model,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: List[float],
    guidance: float = 4.0,
    img_cond: Optional[Tensor] = None,
    img_cond_seq: Optional[Tensor] = None,
    img_cond_seq_ids: Optional[Tensor] = None,
) -> tuple[Tensor, List[Tensor], List[float]]:
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    intermediates = [img.clone()]
    timestep_values = [timesteps[0]]

    def prepare_model_input(current_img: Tensor, current_t: float) -> dict:
        img_input = current_img
        img_input_ids = img_ids
        if img_cond is not None:
            img_input = torch.cat((img_input, img_cond), dim=-1)
        if img_cond_seq is not None:
            assert img_cond_seq_ids is not None, "You need to provide either both or neither of the sequence conditioning"
            img_input = torch.cat((img_input, img_cond_seq), dim=1)
            img_input_ids = torch.cat((img_input_ids, img_cond_seq_ids), dim=1)
        t_vec = torch.full((img.shape[0],), current_t, dtype=img.dtype, device=img.device)
        return {
            "img": img_input,
            "img_ids": img_input_ids,
            "txt": txt,
            "txt_ids": txt_ids,
            "y": vec,
            "timesteps": t_vec,
            "guidance": guidance_vec,
        }

    x = img.clone()
    for i in range(len(timesteps) - 1):
        t_curr = timesteps[i]
        t_prev = timesteps[i + 1]
        h = t_prev - t_curr

        model_input = prepare_model_input(x, t_curr)
        pred_1 = model(**model_input)
        if img_cond_seq_ids is not None:
            pred_1 = pred_1[:, :img.shape[1]]

        t_mid = t_curr + 0.5 * h
        x_mid = x + 0.5 * h * pred_1
        model_input = prepare_model_input(x_mid, t_mid)
        pred_2 = model(**model_input)
        if img_cond_seq_ids is not None:
            pred_2 = pred_2[:, :img.shape[1]]

        x = x + h * pred_2
        intermediates.append(x.clone())
        timestep_values.append(t_prev)

    return x, intermediates, timestep_values
