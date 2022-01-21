def decode_batch_ids(tokenizer, batch_ids):
    """
    Convert batch of token ids to a list of text strings.
    """
    batch_ids_copy = batch_ids.clone()
    batch_ids_copy[batch_ids==-100] = 0
    return [tokenizer.decode(ids,
                             skip_special_tokens=False,
                             clean_up_tokenization_spaces=False)
            for ids in batch_ids_copy]