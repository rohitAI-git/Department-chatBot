"""
main launcher v3.1
"""
import threading
import time
import webbrowser
import uvicorn
hosturl = "http://127.0.0.1:8000"
def openbrowser():
    time.sleep(1.5)  #delaytostart
    webbrowser.open(hosturl)
if __name__ == "__main__":
    threading.Thread(target=openbrowser, daemon=True).start()
    print(f"starting DCSA helpdesk backend at {hosturl} ...")
    uvicorn.run("bkend.app:app", host="127.0.0.1", port=8000, reload=True)