enum class GenerationError
{
    None,
    SequenceLengthExceeded,
    KVCacheExceeded,
    InvalidPrompt,
    Canceled,
    GPUQueueTimedOut
};