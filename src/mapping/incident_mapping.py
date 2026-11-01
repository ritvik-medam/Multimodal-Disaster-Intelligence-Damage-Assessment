"""
Geospatial Mapping Module
Builds interactive Folium maps using only validated coordinates.
Never invents coordinates or places incidents at arbitrary locations.
"""
from typing import Dict, Any, List, Optional
from src.utils.logging_config import get_logger

logger = get_logger("mapping.incident_mapping")

# Priority → marker color mapping
PRIORITY_COLORS = {
    "P1_CRITICAL": "red",
    "P2_HIGH":     "orange",
    "P3_MODERATE": "blue",
    "P4_LOW":      "green",
    "UNASSESSED":  "gray",
}

PRIORITY_ICONS = {
    "P1_CRITICAL": "exclamation-sign",
    "P2_HIGH":     "warning-sign",
    "P3_MODERATE": "info-sign",
    "P4_LOW":      "ok-sign",
    "UNASSESSED":  "question-sign",
}


def build_incident_map(
    incidents: List[Dict[str, Any]],
    selected_categories: Optional[List[str]] = None,
    selected_priorities: Optional[List[str]] = None,
    center_lat: float = 20.0,
    center_lon: float = 0.0,
    zoom_start: int = 2,
) -> Optional[Any]:
    """
    Build a Folium map with incident markers.

    Only incidents with valid lat/lon are placed on the map.
    Incidents without coordinates appear in the dashboard but NOT on the map.

    Args:
        incidents:             List of incident dicts (from repository)
        selected_categories:  Optional filter list for disaster types
        selected_priorities:  Optional filter list for priorities
        center_lat/lon:       Map center
        zoom_start:           Initial zoom level

    Returns:
        folium.Map object, or None if folium is unavailable.
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.error("Folium not installed. Run: pip install folium streamlit-folium")
        return None

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="OpenStreetMap",
    )

    cluster = MarkerCluster(name="Incidents").add_to(m)
    placed  = 0
    skipped = 0

    for inc in incidents:
        lat = inc.get("latitude")
        lon = inc.get("longitude")

        if lat is None or lon is None:
            skipped += 1
            continue

        # Filter by category
        if selected_categories and inc.get("disaster_type") not in selected_categories:
            continue
        # Filter by priority
        if selected_priorities and inc.get("priority") not in selected_priorities:
            continue

        priority = inc.get("priority") or "UNASSESSED"
        color    = PRIORITY_COLORS.get(priority, "gray")
        icon     = PRIORITY_ICONS.get(priority, "info-sign")

        disaster_type = inc.get("disaster_type") or "Unknown"
        title         = inc.get("title") or f"Incident {str(inc.get('incident_id', ''))[:8]}"
        location      = inc.get("reported_location") or "No location text"
        created       = str(inc.get("created_at") or "")[:10]

        type_str = str(disaster_type).replace('_', ' ').title()

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin:0; color:#333;">{title}</h4>
            <hr style="margin:4px 0;">
            <b>Type:</b> {type_str}<br>
            <b>Priority:</b> <span style="color:{_priority_html_color(priority)};">
                {priority}</span><br>
            <b>Location:</b> {location}<br>
            <b>Date:</b> {created}<br>
            <b>ID:</b> {inc.get('incident_id','')}<br>
        </div>
        """

        tooltip = f"{priority} | {disaster_type} | {str(title)[:30]}"

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=tooltip,
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(cluster)
        placed += 1

    if skipped > 0:
        logger.info(
            f"Map: {placed} incidents placed, {skipped} skipped (no valid coordinates)."
        )

    # Legend
    legend_html = _build_legend_html(placed, skipped)
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def _priority_html_color(priority: str) -> str:
    colors = {
        "P1_CRITICAL": "#d32f2f",
        "P2_HIGH": "#e65100",
        "P3_MODERATE": "#1565c0",
        "P4_LOW": "#2e7d32",
        "UNASSESSED": "#616161",
    }
    return colors.get(priority, "#616161")


def _build_legend_html(placed: int, skipped: int) -> str:
    return f"""
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
         background-color: white; padding: 15px; border-radius: 8px;
         border: 2px solid #ccc; font-family: Arial; font-size: 13px;">
        <b>Priority Legend</b><br>
        <span style="color:#d32f2f;">● P1 Critical</span><br>
        <span style="color:#e65100;">● P2 High</span><br>
        <span style="color:#1565c0;">● P3 Moderate</span><br>
        <span style="color:#2e7d32;">● P4 Low</span><br>
        <span style="color:#616161;">● Unassessed</span><br>
        <hr style="margin:6px 0;">
        <small>Showing: {placed} incidents<br>
        No coords (not shown): {skipped}</small>
    </div>
    """


def validate_coordinates_for_map(lat: Optional[float], lon: Optional[float]) -> Dict:
    """
    Validate coordinates before storing or displaying.
    Returns status dict.
    """
    if lat is None and lon is None:
        return {"valid": False, "reason": "No coordinates provided", "will_appear_on_map": False}

    if lat is None or lon is None:
        return {"valid": False, "reason": "Partial coordinates — both lat and lon required",
                "will_appear_on_map": False}

    if not (-90 <= lat <= 90):
        return {"valid": False, "reason": f"Latitude {lat} out of range [-90, 90]",
                "will_appear_on_map": False}

    if not (-180 <= lon <= 180):
        return {"valid": False, "reason": f"Longitude {lon} out of range [-180, 180]",
                "will_appear_on_map": False}

    return {"valid": True, "reason": "Valid coordinates", "will_appear_on_map": True}
