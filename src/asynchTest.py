import asyncio
import os
import shutil
import json
import logging
import csv
from datetime import datetime, timedelta
from inference_sdk import InferenceHTTPClient
from aiohttp import web
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Initialize client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="cMV0aycGJ6Zn2qqycpP6"
)


class ImageProcessor:
    def __init__(self):
        # Image directories
        self.base_dir = "images"
        self.new_dir = os.path.join(self.base_dir, "new")
        self.processing_dir = os.path.join(self.base_dir, "processing")
        self.results_dir = os.path.join(self.base_dir, "results")

        # Thermal data directories
        self.thermal_base_dir = "thermal_data"
        self.thermal_new_dir = os.path.join(self.thermal_base_dir, "new")
        self.thermal_results_dir = os.path.join(self.thermal_base_dir, "results")

        # Final results directory for reports
        self.final_results_dir = "final_results"

        self._create_dirs()
        self._init_report()

    def _create_dirs(self):
        """Create required directories"""
        # Image directories
        for d in [self.new_dir, self.processing_dir, self.results_dir]:
            os.makedirs(d, exist_ok=True)

        # Thermal directories
        for d in [self.thermal_new_dir, self.thermal_results_dir]:
            os.makedirs(d, exist_ok=True)

        # Final results directory
        os.makedirs(self.final_results_dir, exist_ok=True)

        # Error directories
        os.makedirs(os.path.join(self.base_dir, "errors"), exist_ok=True)
        os.makedirs(os.path.join(self.thermal_base_dir, "errors"), exist_ok=True)

    def _init_report(self):
        """Initialize the report file with headers"""
        self.report_path = os.path.join(self.final_results_dir, "temperature_report.csv")
        if not os.path.exists(self.report_path):
            with open(self.report_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Date", "Time", "Image", "Temperature Value"])

    def _add_to_report(self, image_filename, temperature):
        """Add a new entry to the report"""
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        with open(self.report_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([date_str, time_str, image_filename, temperature])
        logging.info(f"Added to report: {image_filename} with temp {temperature}")

    async def process_single_image(self, image_path: str):
        """Process a single image with thermal data pairing"""
        try:
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            thermal_json_path = os.path.join(self.thermal_new_dir, f"{base_name}.json")

            # Check if thermal data exists
            if not os.path.exists(thermal_json_path):
                logging.warning(f"No thermal data found for {image_path}")
                os.remove(image_path)
                return None

            # Run inference
            result = await asyncio.to_thread(
                CLIENT.infer,
                image_path,
                model_id="detection-t88yn/1"
            )

            # Get first prediction class
            first_class = "No prediction"
            if result.get('predictions'):
                first_class = result['predictions'][0].get('class', 'Unknown')
                logging.info(f"First prediction class: {first_class}")

            # Load thermal data
            with open(thermal_json_path, 'r') as f:
                thermal_data = json.load(f)

            # Extract max temperature (adjust key as needed)
            max_temp = thermal_data.get('max_temp', thermal_data.get('temperature', 'N/A'))

            # Create summary
            summary = f"[Label: {first_class}, Max-Temp: {max_temp}]"

            # Process based on label
            if first_class.lower() in ["cat", "dog"]:
                # Save result to file
                result_file = os.path.join(self.results_dir, f"{base_name}_result.txt")
                with open(result_file, "w") as f:
                    f.write(f"Label: {first_class}\n")
                    f.write(f"Max Temperature: {max_temp}\n")
                    f.write(f"Total predictions: {len(result.get('predictions', []))}")

                # Check Temperature Threshold
                try:
                    temp_value = float(max_temp)
                    if temp_value > 38:
                        # Move files to results
                        thermal_result_path = os.path.join(self.thermal_results_dir, f"{base_name}.json")
                        shutil.move(thermal_json_path, thermal_result_path)

                        image_filename = os.path.basename(image_path)
                        image_result_path = os.path.join(self.results_dir, image_filename)
                        shutil.move(image_path, image_result_path)

                        # Add to final report
                        self._add_to_report(image_filename, max_temp)

                        logging.info(f"Kept results: {summary} (Temperature > 38°C)")
                    else:
                        # Delete both files
                        os.remove(image_path)
                        os.remove(thermal_json_path)
                        os.remove(result_file)
                        logging.info(f"Deleted results: {summary} (Temperature <= 38°C)")
                except (ValueError, TypeError):
                    # Handle invalid temperature format
                    logging.error(f"Invalid temperature format: {max_temp}")

                    # Move to errors directory as fallback
                    error_path = os.path.join(self.thermal_base_dir, "errors", f"{base_name}.json")
                    shutil.move(thermal_json_path, error_path)

                    error_img_path = os.path.join(self.base_dir, "errors", os.path.basename(image_path))
                    shutil.move(image_path, error_img_path)

                    logging.warning(f"Moved invalid temperature data to errors: {base_name}")

            else:
                os.remove(image_path)
                os.remove(thermal_json_path)
                logging.info(f"Deleted non-target: {summary}")

            return result

        except Exception as e:
            logging.exception(f"Processing failed for {image_path}")
            # Clean up files on error
            if os.path.exists(image_path):
                error_img_path = os.path.join(self.base_dir, "errors", os.path.basename(image_path))
                shutil.move(image_path, error_img_path)

            thermal_json_path = os.path.join(self.thermal_new_dir,
                                             os.path.splitext(os.path.basename(image_path))[0] + ".json")
            if os.path.exists(thermal_json_path):
                error_thermal_path = os.path.join(self.thermal_base_dir, "errors", os.path.basename(thermal_json_path))
                shutil.move(thermal_json_path, error_thermal_path)

            return None

    async def process_available_images(self, max_concurrent: int = 3):
        """Process images with controlled concurrency and thermal pairing"""
        while True:
            # Get new images with corresponding thermal data
            images = []
            for f in os.listdir(self.new_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    base_name = os.path.splitext(f)[0]
                    thermal_path = os.path.join(self.thermal_new_dir, f"{base_name}.json")
                    if os.path.exists(thermal_path):
                        images.append(os.path.join(self.new_dir, f))

            if not images:
                await asyncio.sleep(0.5)  # Check more frequently
                continue

            # Process batch with concurrency control
            semaphore = asyncio.Semaphore(max_concurrent)

            async def process_with_semaphore(img_path):
                async with semaphore:
                    # Move to processing directory
                    processing_path = os.path.join(self.processing_dir, os.path.basename(img_path))
                    shutil.move(img_path, processing_path)
                    return await self.process_single_image(processing_path)

            await asyncio.gather(*(process_with_semaphore(img) for img in images))

    def take_snapshot(self, image_data: bytes, filename: str):
        """Save new image snapshot"""
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename += ".jpg"

        target_path = os.path.join(self.new_dir, filename)
        with open(target_path, "wb") as f:
            f.write(image_data)
        logging.info(f"New snapshot saved: {target_path}")

    def save_thermal_data(self, thermal_data: dict, filename: str):
        """Save thermal data as JSON"""
        if not filename.lower().endswith('.json'):
            filename += ".json"

        target_path = os.path.join(self.thermal_new_dir, filename)
        with open(target_path, 'w') as f:
            json.dump(thermal_data, f)
        logging.info(f"Thermal data saved: {target_path}")

    def generate_trend_graph(self, filter_type='default'):
        """Generate trend graph as HTML using Plotly with filters"""
        try:
            # Load data from CSV
            df = pd.read_csv(self.report_path)

            # Filter temperatures >37.5°C
            df = df[df['Temperature Value'] > 37.5]

            if df.empty:
                return "<p>No temperature data above 37.5°C available</p>"

            # Create datetime column
            df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

            # Apply filters
            if filter_type == 'daily':
                # Group by date and take max temperature for each day
                df = df.groupby(pd.Grouper(key='DateTime', freq='D')).agg({
                    'Temperature Value': 'max'
                }).reset_index()
                df['Label'] = df['DateTime'].dt.strftime('%Y-%m-%d')
                title = 'Daily Max Fever Trend'
                x_title = 'Date'

            elif filter_type == 'weekly':
                # Group by week and take average for each week
                df['Week'] = df['DateTime'].dt.isocalendar().week
                df['Year'] = df['DateTime'].dt.year
                weekly_avg = df.groupby(['Year', 'Week'])['Temperature Value'].mean().reset_index()
                weekly_avg['Label'] = weekly_avg['Year'].astype(str) + '-W' + weekly_avg['Week'].astype(str)
                df = weekly_avg
                title = 'Weekly Average Fever Trend'
                x_title = 'Week'

            elif filter_type == 'monthly':
                # Group by month and take average for each month
                df['Month'] = df['DateTime'].dt.month
                df['Year'] = df['DateTime'].dt.year
                monthly_avg = df.groupby(['Year', 'Month'])['Temperature Value'].mean().reset_index()
                monthly_avg['Label'] = monthly_avg['Year'].astype(str) + '-' + monthly_avg['Month'].astype(
                    str).str.zfill(2)
                df = monthly_avg
                title = 'Monthly Average Fever Trend'
                x_title = 'Month'

            else:  # default view
                df['Label'] = df['DateTime'].dt.strftime('%Y-%m-%d %H:%M')
                title = 'Fever Trend (Temperatures > 37.5°C)'
                x_title = 'Date and Time'

            # Create the plot
            fig = px.line(
                df,
                x='Label',
                y='Temperature Value',
                title=title,
                labels={'Label': x_title, 'Temperature Value': 'Temperature (°C)'},
                markers=True
            )

            # Customize layout
            fig.update_layout(
                xaxis_title=x_title,
                yaxis_title='Temperature (°C)',
                hovermode='x unified',
                template='plotly_white',
                xaxis=dict(tickangle=45)
            )

            # Convert to HTML
            return pio.to_html(fig, full_html=False)

        except Exception as e:
            logging.error(f"Error generating trend graph: {str(e)}")
            return f"<p>Error generating trend graph: {str(e)}</p>"


class WebServer:
    def __init__(self, processor):
        self.processor = processor
        self.app = web.Application()
        self.setup_routes()

    def setup_routes(self):
        self.app.router.add_get('/', self.handle_home)
        self.app.router.add_get('/thermal-images', self.handle_thermal_images)
        self.app.router.add_get('/thermal-data', self.handle_thermal_data)
        self.app.router.add_get('/report', self.handle_report_download)
        self.app.router.add_static('/results/', path=self.processor.results_dir, show_index=True)
        self.app.router.add_static('/thermal/', path=self.processor.thermal_results_dir, show_index=True)

    async def render_template(self, title, content):
        """Render a common page template"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title} | Thermal Monitoring</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0; 
                    display: flex;
                    min-height: 100vh;
                }}
                .sidebar {{
                    width: 200px;
                    background-color: #f0f0f0;
                    padding: 20px;
                    box-shadow: 2px 0 5px rgba(0,0,0,0.1);
                }}
                .sidebar h2 {{
                    margin-top: 0;
                    color: #333;
                }}
                .sidebar ul {{
                    list-style: none;
                    padding: 0;
                }}
                .sidebar li {{
                    padding: 10px 0;
                    border-bottom: 1px solid #ddd;
                }}
                .sidebar li a {{
                    text-decoration: none;
                    color: #333;
                    display: block;
                }}
                .sidebar li a:hover {{
                    color: #0066cc;
                }}
                .main-content {{
                    flex: 1;
                    padding: 20px;
                }}
                .header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }}
                .header h1 {{
                    margin: 0;
                    color: #333;
                }}
                .tile-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 20px;
                }}
                .tile {{
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .tile img {{
                    width: 100%;
                    height: 150px;
                    object-fit: cover;
                }}
                .tile-content {{
                    padding: 10px;
                }}
                .tile-content h3 {{
                    margin: 0 0 5px;
                    font-size: 16px;
                }}
                .tile-content p {{
                    margin: 5px 0;
                    font-size: 14px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .chart-container {{
                    margin: 20px 0;
                    height: 500px;
                }}
                .filters {{
                    margin: 15px 0;
                    display: flex;
                    gap: 10px;
                    flex-wrap: wrap;
                }}
                .filter-btn {{
                    padding: 8px 15px;
                    background: #f0f0f0;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    cursor: pointer;
                    text-decoration: none;
                    color: #333;
                }}
                .filter-btn.active {{
                    background: #4CAF50;
                    color: white;
                    font-weight: bold;
                }}
                .trend-graph {{
                    border: 1px solid #eee;
                    padding: 15px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }}
                .download-btn {{
                    display: inline-block;
                    padding: 8px 15px;
                    background: #4CAF50;
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    margin-top: 10px;
                }}
                .stats-container {{
                    display: flex;
                    gap: 20px;
                    margin: 20px 0;
                    flex-wrap: wrap;
                }}
                .stat-card {{
                    background: #f9f9f9;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    padding: 15px;
                    min-width: 200px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
            </style>
        </head>
        <body>
            <div class="sidebar">
                <h2>Navigation</h2>
                <ul>
                    <li><a href="/">Home</a></li>
                    <li><a href="/thermal-images">Thermal Images</a></li>
                    <li><a href="/thermal-data">Thermal Data</a></li>
                    <li><a href="/report">Download Report</a></li>
                </ul>
            </div>

            <div class="main-content">
                {content}
            </div>
        </body>
        </html>
        """

    async def handle_home(self, request):
        """Handle home page with tiled images and trend graph"""
        # Get filter from query parameter
        filter_type = request.query.get('filter', 'default')

        # Generate trend graph with selected filter
        graph_html = self.processor.generate_trend_graph(filter_type)

        # Get report data for images
        report_path = self.processor.report_path
        report_data = []
        if os.path.exists(report_path):
            with open(report_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Only include temperatures >37.5°C
                        if float(row['Temperature Value']) > 37.5:
                            report_data.append(row)
                    except (ValueError, TypeError):
                        continue

        # Create tile grid for images
        tiles = []
        for row in report_data:
            img_path = f"/results/{row['Image']}"
            tiles.append(f"""
            <div class="tile">
                <img src="{img_path}" alt="Thermal Image">
                <div class="tile-content">
                    <h3>{row['Image']}</h3>
                    <p><strong>Date:</strong> {row['Date']}</p>
                    <p><strong>Time:</strong> {row['Time']}</p>
                    <p><strong>Temperature:</strong> {row['Temperature Value']}°C</p>
                </div>
            </div>
            """)

        tile_grid = f'<div class="tile-grid">{"".join(tiles)}</div>' if tiles else '<p>No thermal images available</p>'

        # Create filter links with active state
        filters = [
            {'type': 'default', 'label': 'Default View'},
            {'type': 'daily', 'label': 'Daily View'},
            {'type': 'weekly', 'label': 'Weekly View'},
            {'type': 'monthly', 'label': 'Monthly View'}
        ]

        filter_links = []
        for f in filters:
            active = 'active' if f['type'] == filter_type else ''
            filter_links.append(f"""
                <a href="/?filter={f['type']}" class="filter-btn {active}">
                    {f['label']}
                </a>
            """)

        # Calculate statistics
        if report_data:
            temps = [float(r['Temperature Value']) for r in report_data]
            max_temp = max(temps)
            min_temp = min(temps)
            avg_temp = sum(temps) / len(temps)
        else:
            max_temp = min_temp = avg_temp = "N/A"

        # Create statistics cards
        stats_html = f"""
        <div class="stats-container">
            <div class="stat-card">
                <h3>Max Temperature</h3>
                <div class="stat-value">{max_temp}°C</div>
            </div>
            <div class="stat-card">
                <h3>Min Temperature</h3>
                <div class="stat-value">{min_temp}°C</div>
            </div>
            <div class="stat-card">
                <h3>Average Temperature</h3>
                <div class="stat-value">{avg_temp if isinstance(avg_temp, str) else round(avg_temp, 2)}°C</div>
            </div>
            <div class="stat-card">
                <h3>Total Readings</h3>
                <div class="stat-value">{len(report_data)}</div>
            </div>
        </div>
        """ if report_data else ""

        # Trend graph section
        trend_graph = f"""
        <div class="trend-graph">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2>Fever Trend (Temperatures > 37.5°C)</h2>
                <a href="/report" class="download-btn">Download Full Report</a>
            </div>
            <div class="filters">
                {"".join(filter_links)}
            </div>
            <div class="chart-container">
                {graph_html}
            </div>
        </div>
        """

        content = f"""
        <div class="header">
            <h1>Fever Trends Dashboard</h1>
        </div>
        {stats_html}
        {trend_graph}
        <h2>Recent Thermal Images</h2>
        {tile_grid}
        """

        return web.Response(
            text=await self.render_template("Home", content),
            content_type='text/html'
        )

    async def handle_thermal_images(self, request):
        """Handle thermal images page with table view"""
        report_path = self.processor.report_path
        report_data = []
        if os.path.exists(report_path):
            with open(report_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Only include temperatures >37.5°C
                        if float(row['Temperature Value']) > 37.5:
                            report_data.append(row)
                    except (ValueError, TypeError):
                        continue

        # Create table for thermal images
        rows = []
        for row in report_data:
            img_path = f"/results/{row['Image']}"
            rows.append(f"""
            <tr>
                <td>{row['Date']}</td>
                <td>{row['Time']}</td>
                <td><a href="{img_path}">{row['Image']}</a></td>
                <td>{row['Temperature Value']}°C</td>
            </tr>
            """)

        table_content = "\n".join(rows) if rows else '<tr><td colspan="4">No thermal images available</td></tr>'

        content = f"""
        <div class="header">
            <h1>Thermal Images</h1>
            <a href="/report" class="download-btn">Download Full Report</a>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Image</th>
                    <th>Temperature</th>
                </tr>
            </thead>
            <tbody>
                {table_content}
            </tbody>
        </table>
        """

        return web.Response(
            text=await self.render_template("Thermal Images", content),
            content_type='text/html'
        )

    async def handle_thermal_data(self, request):
        """Handle thermal data page with table view"""
        report_path = self.processor.report_path
        report_data = []
        if os.path.exists(report_path):
            with open(report_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        # Only include temperatures >37.5°C
                        if float(row['Temperature Value']) > 37.5:
                            report_data.append(row)
                    except (ValueError, TypeError):
                        continue

        # Create table for thermal data
        rows = []
        for row in report_data:
            # Create thermal data file name
            base_name = os.path.splitext(row['Image'])[0]
            thermal_file = f"{base_name}.json"
            thermal_path = f"/thermal/{thermal_file}"

            rows.append(f"""
            <tr>
                <td>{row['Date']}</td>
                <td>{row['Time']}</td>
                <td><a href="{thermal_path}">{thermal_file}</a></td>
                <td>{row['Temperature Value']}°C</td>
            </tr>
            """)

        table_content = "\n".join(rows) if rows else '<tr><td colspan="4">No thermal data available</td></tr>'

        content = f"""
        <div class="header">
            <h1>Thermal Data</h1>
            <a href="/report" class="download-btn">Download Full Report</a>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Date</th>
                    <th>Time</th>
                    <th>Thermal Data File</th>
                    <th>Temperature</th>
                </tr>
            </thead>
            <tbody>
                {table_content}
            </tbody>
        </table>
        """

        return web.Response(
            text=await self.render_template("Thermal Data", content),
            content_type='text/html'
        )

    async def handle_report_download(self, request):
        """Handle report download"""
        report_path = self.processor.report_path
        if not os.path.exists(report_path):
            return web.Response(text="Report not available", status=404)

        # Create CSV response
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename="temperature_report.csv"'
            }
        )

        await response.prepare(request)

        # Stream the CSV file
        with open(report_path, 'rb') as f:
            while chunk := f.read(4096):
                await response.write(chunk)

        return response

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', 8080)
        await site.start()
        logging.info("Web server running at http://localhost:8080")


async def main():
    processor = ImageProcessor()
    webserver = WebServer(processor)

    # Start processing and web server concurrently
    await asyncio.gather(
        processor.process_available_images(max_concurrent=4),
        webserver.start()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Processing stopped")