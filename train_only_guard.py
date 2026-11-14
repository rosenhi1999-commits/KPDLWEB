"""Audit guard for train-only research: deny opening any blind-named CSV."""
import os,sys

def _audit(event,args):
    if event!='open' or not args:return
    path=args[0]
    if isinstance(path,(str,bytes,os.PathLike)):
        name=os.path.basename(os.fsdecode(path)).lower()
        if name.endswith('.csv') and 'blind' in name:
            raise PermissionError('Train-only mode: opening blind CSV files is prohibited')

sys.addaudithook(_audit)
