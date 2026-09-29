# AcuTrack Multi-Laptop Live Demo Instructions

To present a live demo of AcuTrack to your teacher using 3 laptops as camera inputs, follow these steps:

## Step 1: Connect Laptops to the Same Network
Make sure all 3 laptops are connected to the **same Wi-Fi network** (or hotspot).

## Step 2: Configure and Start the Central Server (Laptop 1)
1. Find Laptop 1's local IP address (e.g., `192.168.1.15`):
   - On Windows, open command prompt and run `ipconfig`. Look for the "IPv4 Address".
2. Open `app.py` on Laptop 1.
3. Change the `LIVE_DEMO_MODE` flag at the top of `app.py` (around line 118) to `True`:
   ```python
   LIVE_DEMO_MODE = True
   ```
4. Start the Flask server on Laptop 1:
   ```bash
   python app.py
   ```
5. Open your browser on Laptop 1 and navigate to the dashboard at `http://localhost:5000`.

## Step 3: Start Camera Streams on Remote Laptops (Laptop 2 and Laptop 3)
1. Copy the `live_camera_client.py` file to Laptop 2 and Laptop 3.
2. Install the required packages on Laptop 2 and Laptop 3:
   ```bash
   pip install opencv-python requests
   ```
3. Run the client on **Laptop 2** to stream to Camera 2:
   ```bash
   python live_camera_client.py --server <Laptop_1_IP> --camera cam2
   ```
   *(Replace `<Laptop_1_IP>` with the actual IP address of Laptop 1, e.g., `192.168.1.15`)*

4. Run the client on **Laptop 3** to stream to Camera 3:
   ```bash
   python live_camera_client.py --server <Laptop_1_IP> --camera cam3
   ```

## Step 4: Run the Live Demo
1. On Laptop 1's dashboard, click **Reset Database** to start with a clean slate.
2. Have a person walk in front of **Laptop 1** (Camera 1). They will be detected and registered.
3. Click the **⚑ Flag** button next to their GID in the sidebar to mark them as a flagged/suspicious target.
4. Have the person walk away from Laptop 1's camera view.
   - The dashboard will show their status in the sidebar as **Not Present** (greyed out).
5. Have the person walk in front of **Laptop 2** (Camera 2) or **Laptop 3** (Camera 3).
   - The dashboard will immediately detect them, match their identity, trigger a browser alert notification, and update their location status to **On Camera 2 (Remote)** or **On Camera 3 (Remote)**!
