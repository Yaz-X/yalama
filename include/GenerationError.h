enum class GenerationError
{
    None,
    SequenceLengthExceeded,
    KVCacheExceeded,
    InvalidPrompt
};