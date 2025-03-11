"""Class to hold all light accessories."""

from __future__ import annotations

import asyncio
import base64
import datetime
from datetime import datetime
import enum
import logging
import time
from typing import Any
import uuid

import tlv8

from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_BRIGHTNESS_PCT,
    ATTR_COLOR_MODE,
    ATTR_COLOR_TEMP_KELVIN,
    ATTR_HS_COLOR,
    ATTR_MAX_COLOR_TEMP_KELVIN,
    ATTR_MIN_COLOR_TEMP_KELVIN,
    ATTR_RGBW_COLOR,
    ATTR_RGBWW_COLOR,
    ATTR_SUPPORTED_COLOR_MODES,
    ATTR_WHITE,
    DOMAIN,
    ColorMode,
    brightness_supported,
    color_supported,
    color_temp_supported,
)
from homeassistant.const import (
    ATTR_ENTITY_ID,
    SERVICE_TURN_OFF,
    SERVICE_TURN_ON,
    STATE_ON,
)
from homeassistant.core import CALLBACK_TYPE, Event, State, callback
from homeassistant.helpers.event import async_call_later, async_track_state_change_event
from homeassistant.helpers.storage import Store
from homeassistant.util.color import (
    color_temperature_kelvin_to_mired,
    color_temperature_mired_to_kelvin,
    color_temperature_to_hs,
    color_temperature_to_rgbww,
)
from pyhap import tlv
from pyhap.const import CATEGORY_LIGHTBULB, HAP_REPR_IID
from pyhap.util import base64_to_bytes, to_base64_str

from .accessories import TYPES, HomeAccessory
from .const import (
    CHAR_BRIGHTNESS,
    CHAR_COLOR_TEMPERATURE,
    CHAR_HUE,
    CHAR_ON,
    CHAR_SATURATION,
    PROP_MAX_VALUE,
    PROP_MIN_VALUE,
    SERV_LIGHTBULB,
)

SUPPORTED_TRANSITION_CONFIGURATION = (
    b"\x01"  # Tag for supported transition configuration
)
CHARACTERISTIC_IID = b"\x01"  # Tag for characteristic instance ID
TRANSITION_TYPE = b"\x02"  # Tag for transition type
BRIGHTNESS = b"\x01"  # Value for brightness transition type
COLOR_TEMPERATURE = b"\x02"  # Value for color temperature transition type

EPOCH_MILLIS_2001_01_01 = int(
    datetime.datetime(2001, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc).timestamp()
    * 1000
)


def bytes_to_base64_string(value: bytes) -> str:
    return base64.b64encode(value).decode("ASCII")


####

_LOGGER = logging.getLogger(__name__)


CHANGE_COALESCE_TIME_WINDOW = 0.01

DEFAULT_MIN_COLOR_TEMP = 2000  # 500 mireds
DEFAULT_MAX_COLOR_TEMP = 7142  # 153 mireds

COLOR_MODES_WITH_WHITES = {ColorMode.RGBW, ColorMode.RGBWW, ColorMode.WHITE}


class ValueTransitionConfigurationResponseTypes(enum.IntEnum):
    VALUE_CONFIGURATION_STATUS = 1


class TransitionControlTypes(enum.IntEnum):
    READ_CURRENT_VALUE_TRANSITION_CONFIGURATION = 1
    UPDATE_VALUE_TRANSITION_CONFIGURATION = 2


class AdaptiveLightingControllerMode(enum.IntEnum):
    AUTOMATIC = 1
    MANUAL = 2


class ValueTransitionParametersTypes(enum.IntEnum):
    START_TIME = 2
    TRANSITION_ID = 1
    UNKNOWN_3 = 3


class ValueTransitionConfigurationStatusTypes(enum.IntEnum):
    CHARACTERISTIC_IID = 1
    TRANSITION_PARAMETERS = 2
    TIME_SINCE_START = 3


class BrightnessAdjustmentMultiplierRange:
    minBrightnessValue = None
    maxBrightnessValue = None


AdaptiveLightingTransitionCurveEntry = {
    "temperature": None,
    "brightness_adjustment_factor": None,
    "transition_time": None,
    "duration": None,
}


class TransitionCurveConfigurationTypes(enum.IntEnum):
    TRANSITION_ENTRY = 1
    ADJUSTMENT_CHARACTERISTIC_IID = 2
    ADJUSTMENT_MULTIPLIER_RANGE = 3


class ActiveAdaptiveLightingTransition:
    iid = 0
    transition_start_millis = 0
    time_millis_offset = 0
    transition_id = None
    transition_start_buffer = None
    transition_curve = AdaptiveLightingTransitionCurveEntry
    brightness_characteristic_IID = None
    brightness_adjustment_range = BrightnessAdjustmentMultiplierRange
    update_interval = None
    notify_interval_threshold = None


def epochMillisFromMillisSince2001_01_01(millis):
    return EPOCH_MILLIS_2001_01_01 + millis


def epochMillisFromMillisSince2001_01_01Buffer(millis):
    millisince2001 = tlv.read_uint64_le(millis)
    return epochMillisFromMillisSince2001_01_01(millisince2001)


@TYPES.register("Light")
class Light(HomeAccessory):
    """Generate a Light accessory for a light entity.

    Currently supports: state, brightness, color temperature, rgb_color.
    """

    STORAGE_KEY = "adaptive_lighting_state"

    def __init__(self, *args: Any) -> None:
        """Initialize a new Light accessory object."""
        super().__init__(*args, category=CATEGORY_LIGHTBULB)
        self.storage = Store(self.hass, version=1, key=self.STORAGE_KEY)
        self.adaptive_lighting_active = False  # Default state
        self.active_transition = None  # Default transition state
        self.char_tc_value = None  # Default TransitionControl value
        self.initial_transition_control_value = None  # Save the initial set_tc value

        # Load the stored state on initialization
        self.hass.loop.create_task(self._load_state())
        self.last_transition_point_info = None
        self._reload_on_change_attrs.extend(
            (
                ATTR_SUPPORTED_COLOR_MODES,
                ATTR_MAX_COLOR_TEMP_KELVIN,
                ATTR_MIN_COLOR_TEMP_KELVIN,
            )
        )
        self.chars = [
            "Brightness",
            "ColorTemperature",
            "ActiveTransitionCount",
            "TransitionControl",
            "SupportedTransitionConfiguration",
        ]
        self._event_timer: CALLBACK_TYPE | None = None
        self._pending_events: dict[str, Any] = {}

        state = self.hass.states.get(self.entity_id)
        assert state
        attributes = state.attributes
        self.color_modes = color_modes = (
            attributes.get(ATTR_SUPPORTED_COLOR_MODES) or []
        )
        self._previous_color_mode = attributes.get(ATTR_COLOR_MODE)
        self.color_supported = color_supported(color_modes)
        self.color_temp_supported = color_temp_supported(color_modes)
        self.rgbw_supported = ColorMode.RGBW in color_modes
        self.rgbww_supported = ColorMode.RGBWW in color_modes
        self.white_supported = ColorMode.WHITE in color_modes
        self.brightness_supported = brightness_supported(color_modes)
        # self.active_transition = ActiveAdaptiveLightingTransition
        self.active_transition = None
        self.mode = AdaptiveLightingControllerMode
        self.update_timeout = None
        self.handle_adjustment_factor_changed = (
            self.handle_adjustment_factor_changed_listner
        )
        self.adaptive_lighting_active = False
        self.did_run_first_initialization_step = False

        if self.brightness_supported:
            self.chars.append(CHAR_BRIGHTNESS)

        if self.color_supported:
            self.chars.extend([CHAR_HUE, CHAR_SATURATION])

        if self.color_temp_supported or COLOR_MODES_WITH_WHITES.intersection(
            self.color_modes
        ):
            self.chars.append(CHAR_COLOR_TEMPERATURE)

        serv_light = self.add_preload_service(SERV_LIGHTBULB, self.chars)
        self.char_on = serv_light.configure_char(CHAR_ON, value=0)

        if self.brightness_supported:
            self.char_brightness = serv_light.configure_char(CHAR_BRIGHTNESS, value=100)

        if CHAR_COLOR_TEMPERATURE in self.chars:
            self.min_mireds = color_temperature_kelvin_to_mired(
                attributes.get(ATTR_MAX_COLOR_TEMP_KELVIN, DEFAULT_MAX_COLOR_TEMP)
            )
            self.max_mireds = color_temperature_kelvin_to_mired(
                attributes.get(ATTR_MIN_COLOR_TEMP_KELVIN, DEFAULT_MIN_COLOR_TEMP)
            )
            if not self.color_temp_supported and not self.rgbww_supported:
                self.max_mireds = self.min_mireds
            self.char_color_temp = serv_light.configure_char(
                CHAR_COLOR_TEMPERATURE,
                value=self.min_mireds,
                properties={
                    PROP_MIN_VALUE: self.min_mireds,
                    PROP_MAX_VALUE: self.max_mireds,
                },
            )

        if self.color_supported:
            self.char_hue = serv_light.configure_char(CHAR_HUE, value=0)
            self.char_saturation = serv_light.configure_char(CHAR_SATURATION, value=75)

        # Simulate the instance ID retrieval
        brightness_iid = self.char_brightness.to_HAP()[HAP_REPR_IID].to_bytes(
            2, "little"
        )
        temperature_iid = self.char_color_temp.to_HAP()[HAP_REPR_IID].to_bytes(
            2, "little"
        )

        # Encode transitions in TLV format using the encode function from tlv.py
        encoded_data = tlv.encode(
            SUPPORTED_TRANSITION_CONFIGURATION,
            tlv.encode(
                CHARACTERISTIC_IID,
                brightness_iid,
                TRANSITION_TYPE,
                BRIGHTNESS,
                CHARACTERISTIC_IID,
                temperature_iid,
                TRANSITION_TYPE,
                COLOR_TEMPERATURE,
            ),
        )

        # Convert to base64 for output
        b64str = base64.b64encode(encoded_data).decode("utf-8")
        self.char_br = serv_light.configure_char(
            "Brightness", setter_callback=self.set_brightness
        )
        self.char_ct = serv_light.configure_char(
            "ColorTemperature", setter_callback=self.set_ct, value=140
        )

        self.char_atc = serv_light.configure_char(
            "ActiveTransitionCount", setter_callback=self.set_atc
        )
        self.char_tc = serv_light.configure_char(
            "TransitionControl", setter_callback=self.set_tc
        )
        self.char_stc = serv_light.configure_char(
            "SupportedTransitionConfiguration", value=b64str
        )
        self.async_update_state(state)
        serv_light.setter_callback = self._set_chars

    async def _load_state(self):
        """Load the adaptive lighting state, active transition, and characteristic value."""
        data = await self.storage.async_load()
        if data:
            self.adaptive_lighting_active = data.get("adaptive_lighting_active", False)
            self.active_transition = data.get("active_transition", None)
            self.transition_control_value = data.get("transition_control_value", None)

            if self.adaptive_lighting_active and self.active_transition:
                _LOGGER.info("Restoring adaptive lighting from saved state.")
                self.restore_active_transition()

            if len(self.transition_control_value) > 15:
                _LOGGER.info("Restoring TransitionControl characteristic value.")
                self.char_tc.set_value(self.transition_control_value)

    async def _save_state(self):
        """Save the adaptive lighting state, active transition, and characteristic value."""
        await self.storage.async_save(
            {
                "adaptive_lighting_active": self.adaptive_lighting_active,
                "active_transition": self.active_transition,
                "transition_control_value": self.transition_control_value,
            }
        )

    def set_ct(self, value):
        """Set color temperature of the light."""
        logging.info("%s Bulb color temp: %s", self.entity_id, value)
        self.char_ct.set_value(value)
        self.char_ct.notify()
        params = {
            ATTR_ENTITY_ID: self.entity_id,
            ATTR_COLOR_TEMP_KELVIN: color_temperature_mired_to_kelvin(value),
        }
        self.async_call_service(DOMAIN, SERVICE_TURN_ON, params)

    def set_brightness(self, value):
        """Set brightness of the light."""
        logging.info("Bulb brightness: %s", value)
        self.char_br.set_value(value)
        self.char_br.notify()

        params = {
            ATTR_ENTITY_ID: self.entity_id,
            ATTR_BRIGHTNESS: value / 100 * 255,
        }
        self.async_call_service(DOMAIN, SERVICE_TURN_ON, params)

    def set_atc(self, now):
        now = int(round(datetime.datetime.now().timestamp() * 1000))
        if not self.active_transition:
            return b""
        active = self.active_transition
        time_since_start = (
            now
            if now is None
            else (
                now - active["time_millis_offset"] - active["transition_start_millis"]
            )
        )

        time_since_start_buffer = tlv.write_variable_uint_le(time_since_start)
        parameters = tlv8.encode(
            [
                tlv8.Entry(
                    ValueTransitionParametersTypes.TRANSITION_ID,
                    bytes.fromhex(active["transition_id"].replace("-", "")),
                ),
                tlv8.Entry(
                    ValueTransitionParametersTypes.START_TIME,
                    base64_to_bytes(active["transition_start_buffer"]),
                ),
            ]
        )
        if active["id3"]:
            parameters += tlv8.encode(
                [
                    tlv8.Entry(
                        ValueTransitionParametersTypes.UNKNOWN_3,
                        base64_to_bytes(active["id3"]),
                    )
                ]
            )
        status = tlv8.encode(
            [
                tlv8.Entry(
                    ValueTransitionConfigurationStatusTypes.CHARACTERISTIC_IID,
                    tlv.write_variable_uint_le(active["iid"]),
                ),
                tlv8.Entry(
                    ValueTransitionConfigurationStatusTypes.TRANSITION_PARAMETERS,
                    parameters,
                ),
                tlv8.Entry(
                    ValueTransitionConfigurationStatusTypes.TIME_SINCE_START,
                    time_since_start_buffer,
                ),
            ]
        )

        return tlv8.encode(
            [
                tlv8.Entry(
                    ValueTransitionConfigurationResponseTypes.VALUE_CONFIGURATION_STATUS,
                    status,
                )
            ]
        )

    def set_tc(self, value):
        """Set TransitionControl characteristic."""
        _LOGGER.info("Write to Transition Control: %s", value)

        # Save the initial value of set_tc
        if self.initial_transition_control_value is None:
            self.initial_transition_control_value = value
            _LOGGER.info("Saved initial Transition Control value: %s", value)

        self.transition_control_value = value
        self.hass.loop.create_task(self._save_state())
        self.transition_control_value = value
        self.hass.loop.create_task(
            self._save_state()
        )  # Save state whenever this is set
        # Decode the value, process it, and store it
        tlv_data = tlv.decode(base64_to_bytes(value))
        response_buffers = []

        read_transition = None
        if read_transition:
            read_transition_response = self.handle_transition_control_read_transition(
                read_transition
            )
            if read_transition_response:
                response_buffers.append(read_transition_response)

        update_transition = tlv_data.get(b"\x02")
        if update_transition:
            update_transition_response = (
                self.handle_transition_control_update_transition(update_transition)
            )
            if update_transition_response:
                response_buffers.append(update_transition_response)

        return to_base64_str(b"".join(response_buffers))

    def handle_transition_control_update_transition(self, buffer):
        update_transition = tlv.decode(buffer)
        transition_configuration = tlv.decode(update_transition[b"\x01"])
        iid = tlv.read_variable_uint_le(transition_configuration[b"\x01"])

        param3 = transition_configuration.get(b"\x03", [None])[
            0
        ]  # when present it is always 1

        if not param3:  # if HomeKit just sends the iid, we consider that as "disable adaptive lighting" (assumption)
            self.handle_adaptive_lighting_disabled()
            return tlv8.encode(
                [
                    tlv8.Entry(
                        TransitionControlTypes.UPDATE_VALUE_TRANSITION_CONFIGURATION,
                        b"",
                    )
                ]
            )
        parameters_tlv = tlv.decode(transition_configuration[b"\x02"])
        curve_configuration = tlv.decode_with_lists(transition_configuration[b"\x05"])
        update_interval = (
            tlv.read_uint16(transition_configuration[b"\x06"])
            if transition_configuration[b"\x06"]
            else None
        )
        notify_interval_threshold = tlv.read_uint32(transition_configuration[b"\x08"])

        transition_id = parameters_tlv[b"\x01"]
        start_time = parameters_tlv[b"\x02"]
        id3 = (
            parameters_tlv[b"\x03"] if parameters_tlv[b"\x03"] else None
        )  # this may be undefined
        start_time_millis = epochMillisFromMillisSince2001_01_01Buffer(start_time)
        time_millis_offset = (
            int(round(datetime.datetime.now().timestamp() * 1000)) - start_time_millis
        )

        transition_curve = []
        previous = None
        transitions = curve_configuration[
            TransitionCurveConfigurationTypes.TRANSITION_ENTRY
        ]

        for idx, entry in enumerate(transitions):
            tlv_entry = tlv.decode(entry)
            adjustment_factor = tlv.read_float_le(tlv_entry[b"\x01"], 0)
            value = tlv.read_float_le(tlv_entry[b"\x02"], 0)
            transition_offset = tlv.read_variable_uint_le(tlv_entry[b"\x03"])
            duration = None

            # Fix transition_time for intermediate curves
            if idx == 0 or idx == len(transitions) - 1:
                transition_time = transition_offset * 28125  # Retain original
            else:
                transition_time = 1800000  # Fixed for all intermediate curves

            if previous:
                previous["duration"] = duration
            previous = {
                "temperature": value,
                "brightness_adjustment_factor": adjustment_factor,
                "transition_time": transition_time,
            }
            transition_curve.append(previous)

        adjustment_iid = tlv.read_variable_uint_le(
            curve_configuration[
                TransitionCurveConfigurationTypes.ADJUSTMENT_CHARACTERISTIC_IID
            ]
        )
        adjustment_multiplier_range = tlv.decode(
            curve_configuration[
                TransitionCurveConfigurationTypes.ADJUSTMENT_MULTIPLIER_RANGE
            ]
        )
        min_adjustment_multiplier = tlv.read_uint32_le(
            adjustment_multiplier_range[b"\x01"], 0
        )
        max_adjustment_multiplier = tlv.read_uint32_le(
            adjustment_multiplier_range[b"\x02"], 0
        )
        self.active_transition = {
            "iid": iid,
            "transition_start_millis": start_time_millis,
            "time_millis_offset": time_millis_offset,
            "transition_id": str(uuid.UUID(bytes=transition_id)),
            "transition_start_buffer": to_base64_str(start_time),
            "id3": to_base64_str(id3) if id3 else None,
            "brightness_characteristic_iid": adjustment_iid,
            "brightness_adjustment_range": {
                "min_brightness_value": min_adjustment_multiplier,
                "max_brightness_value": max_adjustment_multiplier,
            },
            "transition_curve": transition_curve,
            "update_interval": update_interval if update_interval else 60000,
            "notify_interval_threshold": notify_interval_threshold,
        }

        print("transition_curve________________", transition_curve)
        if self.update_timeout:
            self.update_timeout.cancel()
            self.update_timeout = None
            logging.info("Adaptive lighting was renewed.")
        else:
            logging.info("Adaptive lighting was enabled.")

        self.handle_active_transition_updated()
        return tlv8.encode(
            [
                tlv8.Entry(
                    TransitionControlTypes.UPDATE_VALUE_TRANSITION_CONFIGURATION,
                    self.set_atc(0),
                )
            ]
        )

    def handle_active_transition_updated(self):
        """Handle the enabling or updating of adaptive lighting."""
        self.adaptive_lighting_active = True
        self.hass.loop.create_task(self._save_state())  # Save state
        _LOGGER.info("Adaptive lighting enabled and state saved.")
        self.schedule_next_update()

    def handle_adaptive_lighting_disabled(self):
        """Disable adaptive lighting and reset settings."""
        self.adaptive_lighting_active = False
        _LOGGER.info(f"Adaptive lighting disabled for {self.entity_id}.")
        self.active_transition = None
        self.transition_control_value = None  # Corrected variable

        # Clear TransitionControl characteristic value
        if self.char_tc:
            self.char_tc.set_value(None)

        self.hass.loop.create_task(self._save_state())

        if self.char_atc:
            self.char_atc.set_value(0)
            self.char_atc.notify()

        # Reset characteristics to current values without changing physical state
        current_state = self.hass.states.get(self.entity_id)
        if current_state:
            self.async_update_state(current_state)

        # Clear transition state
        self.last_transition_point_info = None
        self.did_run_first_initialization_step = False

        # Cancel any ongoing updates
        if self.update_timeout:
            self.update_timeout.cancel()
            self.update_timeout = None

        _LOGGER.info("Adaptive lighting fully disabled.")

    def restore_active_transition(self):
        """Restore the active transition after restart."""
        if not self.active_transition:
            _LOGGER.warning("No active transition to restore.")
            return

        _LOGGER.info("Restoring active transition.")
        # Reapply the transition details (e.g., timing, curve, etc.)
        self.handle_active_transition_updated()

    def get_current_adaptive_lighting_transition_point(self):
        if not self.active_transition:
            raise ValueError(
                "Cannot calculate the current transition point if no transition is active!"
            )
        adjusted_now = int(
            time.time() * 1000 - self.active_transition["time_millis_offset"]
        )
        offset = adjusted_now - self.active_transition["transition_start_millis"]
        i = (
            self.last_transition_point_info["curve_index"]
            if self.last_transition_point_info
            else 0
        )
        lower_bound_time_offset = (
            self.last_transition_point_info["lower_bound_time_offset"]
            if self.last_transition_point_info
            else 0
        )
        lower_bound = None
        upper_bound = None
        for i in range(i, len(self.active_transition["transition_curve"]) - 1):
            lower_bound0 = self.active_transition["transition_curve"][i]
            upper_bound0 = self.active_transition["transition_curve"][i + 1]
            lower_bound_duration = (
                lower_bound0["duration"] if lower_bound0["duration"] else 0
            )
            lower_bound_time_offset += lower_bound0["transition_time"]
            if offset >= lower_bound_time_offset:
                if (
                    offset
                    <= lower_bound_time_offset
                    + lower_bound_duration
                    + upper_bound0["transition_time"]
                ):
                    lower_bound = lower_bound0
                    upper_bound = upper_bound0
                    break
            elif self.last_transition_point_info:
                self.last_transition_point_info = None
                return self.get_current_adaptive_lighting_transition_point()
            lower_bound_time_offset += lower_bound_duration
        if not lower_bound or not upper_bound:
            self.last_transition_point_info = None
            return None
        self.last_transition_point_info = {
            "curve_index": i,
            "lower_bound_time_offset": lower_bound_time_offset
            - lower_bound["transition_time"],
        }
        print("last_transition ", self.last_transition_point_info)
        print(
            "low_bound",
            {
                "lower_bound_time_offset": lower_bound_time_offset,
                "transition_offset": offset - lower_bound_time_offset,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        )
        return {
            "lower_bound_time_offset": lower_bound_time_offset,
            "transition_offset": offset - lower_bound_time_offset,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
        }

    def schedule_next_update(self, dry_run=False):
        if not self.active_transition:
            raise ValueError(
                "Tried scheduling transition when no transition was active!"
            )
            return

        if not dry_run:
            self.update_timeout = None

        if not self.did_run_first_initialization_step:
            self.did_run_first_initialization_step = True

        transition_point = self.get_current_adaptive_lighting_transition_point()
        print("transition_Point", transition_point)
        if not transition_point:
            _LOGGER.info("Reached end of transition curve! Restarting...")
            # Reset the start time to now to restart the transition
            current_time = int(time.time() * 1000)
            self.active_transition["transition_start_millis"] = (
                current_time - self.active_transition["time_millis_offset"]
            )
            self.last_transition_point_info = None
            # Immediately schedule the next update to restart the transition
            self.schedule_next_update()
            return

        lower_bound = transition_point["lower_bound"]
        upper_bound = transition_point["upper_bound"]

        if (
            lower_bound["duration"]
            and transition_point["transition_offset"] <= lower_bound["duration"]
        ):
            interpolated_temperature = lower_bound["temperature"]
            interpolated_adjustment_factor = lower_bound["brightness_adjustment_factor"]
        else:
            time_percentage = (
                transition_point["transition_offset"]
                - (lower_bound["duration"] if lower_bound["duration"] else 0)
            ) / upper_bound["transition_time"]
            interpolated_temperature = (
                lower_bound["temperature"]
                + (upper_bound["temperature"] - lower_bound["temperature"])
                * time_percentage
            )
            interpolated_adjustment_factor = (
                lower_bound["brightness_adjustment_factor"]
                + (
                    upper_bound["brightness_adjustment_factor"]
                    - lower_bound["brightness_adjustment_factor"]
                )
                * time_percentage
            )

        adjustment_multiplier = max(
            self.active_transition["brightness_adjustment_range"][
                "min_brightness_value"
            ],
            min(
                self.active_transition["brightness_adjustment_range"][
                    "max_brightness_value"
                ],
                self.char_br.get_value(),
            ),
        )
        temperature = round(
            interpolated_temperature
            + interpolated_adjustment_factor * adjustment_multiplier
        )

        min_temp = 140
        max_temp = 500
        temperature = max(min_temp, min(max_temp, temperature))
        print("TEMP__", temperature)
        self.set_ct(temperature)
        self.char_ct.set_value(temperature)
        self.char_ct.notify()

        # Sending event notifications
        now = int(round(datetime.datetime.now().timestamp() * 1000))
        if not dry_run and now >= self.active_transition["notify_interval_threshold"]:
            self.last_event_notification_sent = now
        if not dry_run:
            self.update_timeout = asyncio.get_event_loop().call_later(
                self.active_transition["update_interval"] / 1000,
                self.schedule_next_update,
            )

    def handle_characteristic_manual_written(brightness):
        pass

    def handle_adjustment_factor_changed_listner(self, event: Event):
        async_track_state_change_event(self.schedule_next_update(True))

    def _set_chars(self, char_values: dict[str, Any]) -> None:
        _LOGGER.debug("Light _set_chars: %s", char_values)
        if CHAR_COLOR_TEMPERATURE in self._pending_events and (
            CHAR_SATURATION in char_values or CHAR_HUE in char_values
        ):
            del self._pending_events[CHAR_COLOR_TEMPERATURE]
        for char in (CHAR_HUE, CHAR_SATURATION):
            if char in self._pending_events and CHAR_COLOR_TEMPERATURE in char_values:
                del self._pending_events[char]

        self._pending_events.update(char_values)
        if self._event_timer:
            self._event_timer()
        self._event_timer = async_call_later(
            self.hass, CHANGE_COALESCE_TIME_WINDOW, self._async_send_events
        )

    @callback
    def _async_send_events(self, _now: datetime) -> None:
        """Process all changes at once."""
        _LOGGER.debug("Coalesced _set_chars: %s", self._pending_events)
        char_values = self._pending_events
        self._pending_events = {}
        events = []
        service = SERVICE_TURN_ON
        params: dict[str, Any] = {ATTR_ENTITY_ID: self.entity_id}

        if CHAR_ON in char_values:
            if not char_values[CHAR_ON]:
                service = SERVICE_TURN_OFF
            events.append(f"Set state to {char_values[CHAR_ON]}")

        brightness_pct = None
        if CHAR_BRIGHTNESS in char_values:
            if char_values[CHAR_BRIGHTNESS] == 0:
                events[-1] = "Set state to 0"
                service = SERVICE_TURN_OFF
            else:
                brightness_pct = char_values[CHAR_BRIGHTNESS]
            events.append(f"brightness at {char_values[CHAR_BRIGHTNESS]}%")

        if service == SERVICE_TURN_OFF:
            self.async_call_service(
                DOMAIN, service, {ATTR_ENTITY_ID: self.entity_id}, ", ".join(events)
            )
            return

        # Handle white channels
        if CHAR_COLOR_TEMPERATURE in char_values:
            temp = char_values[CHAR_COLOR_TEMPERATURE]
            events.append(f"color temperature at {temp}")
            bright_val = round(
                ((brightness_pct or self.char_brightness.value) * 255) / 100
            )
            if self.color_temp_supported:
                params[ATTR_COLOR_TEMP_KELVIN] = color_temperature_mired_to_kelvin(temp)
            elif self.rgbww_supported:
                params[ATTR_RGBWW_COLOR] = color_temperature_to_rgbww(
                    color_temperature_mired_to_kelvin(temp),
                    bright_val,
                    color_temperature_mired_to_kelvin(self.max_mireds),
                    color_temperature_mired_to_kelvin(self.min_mireds),
                )
            elif self.rgbw_supported:
                params[ATTR_RGBW_COLOR] = (*(0,) * 3, bright_val)
            elif self.white_supported:
                params[ATTR_WHITE] = bright_val

        elif CHAR_HUE in char_values or CHAR_SATURATION in char_values:
            hue_sat = (
                char_values.get(CHAR_HUE, self.char_hue.value),
                char_values.get(CHAR_SATURATION, self.char_saturation.value),
            )
            _LOGGER.debug("%s: Set hs_color to %s", self.entity_id, hue_sat)
            events.append(f"set color at {hue_sat}")
            params[ATTR_HS_COLOR] = hue_sat

        if (
            brightness_pct
            and ATTR_RGBWW_COLOR not in params
            and ATTR_RGBW_COLOR not in params
        ):
            params[ATTR_BRIGHTNESS_PCT] = brightness_pct

        _LOGGER.debug(
            "Calling light service with params: %s -> %s", char_values, params
        )
        self.async_call_service(DOMAIN, service, params, ", ".join(events))

    @callback
    def async_update_state(self, new_state: State) -> None:
        if not self.adaptive_lighting_active:
            return  # Skip updates if disabled
        """Update light after state change."""
        # Handle State
        state = new_state.state
        attributes = new_state.attributes
        color_mode = attributes.get(ATTR_COLOR_MODE)
        self.char_on.set_value(int(state == STATE_ON))
        color_mode_changed = self._previous_color_mode != color_mode
        self._previous_color_mode = color_mode

        # Handle Brightness
        if (
            self.brightness_supported
            and (brightness := attributes.get(ATTR_BRIGHTNESS)) is not None
            and isinstance(brightness, (int, float))
        ):
            brightness = round(brightness / 255 * 100, 0)

            if brightness == 0 and state == STATE_ON:
                brightness = 1
            self.char_brightness.set_value(brightness)
            if color_mode_changed:
                self.char_brightness.notify()

        # Handle Color - color must always be set before color temperature
        # or the iOS UI will not display it correctly.
        if self.color_supported:
            if color_temp := attributes.get(ATTR_COLOR_TEMP_KELVIN):
                hue, saturation = color_temperature_to_hs(color_temp)
            elif color_mode == ColorMode.WHITE:
                hue, saturation = 0, 0
            elif hue_sat := attributes.get(ATTR_HS_COLOR):
                hue, saturation = hue_sat
            else:
                hue = None
                saturation = None
            if isinstance(hue, (int, float)) and isinstance(saturation, (int, float)):
                self.char_hue.set_value(round(hue, 0))
                self.char_saturation.set_value(round(saturation, 0))
                if color_mode_changed:
                    # If the color temp changed, be sure to force the color to update
                    self.char_hue.notify()
                    self.char_saturation.notify()

        # Handle white channels
        if CHAR_COLOR_TEMPERATURE in self.chars:
            color_temp = None
            if self.color_temp_supported:
                color_temp_kelvin = attributes.get(ATTR_COLOR_TEMP_KELVIN)
                if color_temp_kelvin is not None:
                    color_temp = color_temperature_kelvin_to_mired(color_temp_kelvin)
            elif color_mode == ColorMode.WHITE:
                color_temp = self.min_mireds
            if isinstance(color_temp, (int, float)):
                self.char_color_temp.set_value(round((color_temp), 0))
                if color_mode_changed:
                    self.char_color_temp.notify()
