import EarthSatelliteScene from "./earthSatellite";
import SatelliteVisualization from "./mlvisual";
const Demo = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900">
      <div className="mt-24 w-full max-w-7xl mx-auto px-4 py-8">
        {/* Earth Satellite Scene */}
        <div className="mb-12">
          <h2 className="font-circular-web text-2xl font-semibold text-white text-center mb-6">
            Earth Satellite Visualization
          </h2>
          <div className="rounded-lg overflow-hidden shadow-2xl">
            <EarthSatelliteScene />
          </div>
        </div>

        {/* ML Task Priority Queue Visualization */}
        <div className="mb-8">
          <h2 className="font-circular-web text-2xl font-semibold text-white text-center mb-6">
            ML Task Priority Queue Visualization
          </h2>
          <div className="rounded-lg overflow-hidden shadow-2xl">
            <SatelliteVisualization />
          </div>
        </div>
      </div>
    </div>
  );
};

export default Demo;
