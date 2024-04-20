from datasets import Audio, DatasetDict, load_dataset


def get_common_voice(
    name="mozilla-foundation/common_voice_11_0",
    language="ru",
    train_split="train[0:3000]",
    test_split="test[0:500]",
    sampling_rate=16000,
    removavle_cols=[
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ],
):
    common_voice = DatasetDict()
    common_voice["train"] = load_dataset(name, language, split=train_split)
    common_voice["test"] = load_dataset(name, language, split=test_split)
    common_voice = common_voice.remove_columns(removavle_cols)
    common_voice = common_voice.cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )
    return common_voice


def get_common_voice_subset(
    name="mozilla-foundation/common_voice_11_0",
    language="ru",
    split="train[0:3000]",
    sampling_rate=16000,
    removavle_cols=[
        "accent",
        "age",
        "client_id",
        "down_votes",
        "gender",
        "locale",
        "path",
        "segment",
        "up_votes",
    ],
):
    common_voice = load_dataset(name, language, split=split)
    common_voice = common_voice.remove_columns(removavle_cols)
    common_voice = common_voice.cast_column(
        "audio", Audio(sampling_rate=sampling_rate)
    )
    return common_voice
