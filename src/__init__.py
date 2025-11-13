# Apply huggingface_hub compatibility patch before any pyannote imports
# This fixes the issue where pyannote.audio uses 'use_auth_token' which is
# deprecated in huggingface_hub >= 0.20.0

def _patch_huggingface_hub_compatibility():
    """Patch huggingface_hub to accept use_auth_token for backward compatibility with pyannote.audio."""
    try:
        import huggingface_hub
        import functools
        import sys
        
        # Store original if not already patched
        if not hasattr(huggingface_hub, '_hf_hub_download_patched'):
            original = huggingface_hub.hf_hub_download
            
            @functools.wraps(original)
            def patched_hf_hub_download(*args, **kwargs):
                # Convert use_auth_token to token for compatibility
                if 'use_auth_token' in kwargs:
                    token_val = kwargs.pop('use_auth_token')
                    if 'token' not in kwargs:
                        kwargs['token'] = token_val
                return original(*args, **kwargs)
            
            # Apply patch to the main module
            huggingface_hub.hf_hub_download = patched_hf_hub_download
            huggingface_hub._hf_hub_download_patched = True
            
            # Also patch in utils submodule if it exists and has its own reference
            try:
                if hasattr(huggingface_hub, 'utils'):
                    huggingface_hub.utils.hf_hub_download = patched_hf_hub_download
            except (AttributeError, TypeError):
                pass
            
            # Patch any already-imported modules that might have a reference
            for mod_name in list(sys.modules.keys()):
                if 'huggingface_hub' in mod_name:
                    try:
                        mod = sys.modules[mod_name]
                        if hasattr(mod, 'hf_hub_download') and mod.hf_hub_download is original:
                            setattr(mod, 'hf_hub_download', patched_hf_hub_download)
                    except (AttributeError, TypeError):
                        pass
    except (ImportError, AttributeError):
        pass

# Apply patch immediately when this package is imported
# This must happen before pyannote.audio is imported
_patch_huggingface_hub_compatibility()

