"""Infrastructure failures must not become permanent data-quality rejections."""
import errno


def retryable_resource_error(error):
    codes={getattr(errno,name,None) for name in
           ('ENOSPC','EDQUOT','ENOMEM','EIO','EROFS','ESTALE','EMFILE','ENFILE')}
    codes.discard(None)
    return isinstance(error,MemoryError) or (isinstance(error,OSError) and error.errno in codes)
