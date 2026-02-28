import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { SplitText } from "gsap/all";
import { useRef } from "react";
import { useMediaQuery } from "react-responsive";

const Hero = () => {
	const videoRef = useRef();

	const isMobile = useMediaQuery({ maxWidth: 767 });

	useGSAP(() => {
		const heroSplit = new SplitText(".title", {
			type: "chars, words",
		});

		const paragraphSplit = new SplitText(".subtitle", {
			type: "lines",
		});

		// Apply gradient to subtitle lines to fix visibility issue with SplitText
		paragraphSplit.lines.forEach((line) => line.classList.add("text-gradient"));

		// Restore high-impact theme-aware gradient
		heroSplit.chars.forEach((char) => char.classList.add("text-gradient"));

		gsap.from(heroSplit.chars, {
			yPercent: 100,
			duration: 1.8,
			ease: "expo.out",
			stagger: 0.06,
		});

		gsap.from(paragraphSplit.lines, {
			opacity: 0,
			yPercent: 100,
			duration: 1.8,
			ease: "expo.out",
			stagger: 0.06,
			delay: 1,
		});

		gsap
			.timeline({
				scrollTrigger: {
					trigger: "#hero",
					start: "top top",
					end: "bottom top",
					scrub: true,
				},
			})
			.to(".right-leaf", { y: 200, opacity: 0.2 }, 0)
			.to(".left-leaf", { y: -200, opacity: 0.2 }, 0)
			.to(".arrow", { y: 100, opacity: 0 }, 0);

		const startValue = isMobile ? "top 50%" : "center 60%";
		const endValue = isMobile ? "120% top" : "bottom top";

		let tl = gsap.timeline({
			scrollTrigger: {
				trigger: "video",
				start: startValue,
				end: endValue,
				scrub: true,
				pin: true,
			},
		});

		videoRef.current.onloadedmetadata = () => {
			tl.to(videoRef.current, {
				currentTime: videoRef.current.duration,
			});
		};

		gsap.from(".hero-btn", {
			y: 20,
			duration: 1.5,
			ease: "expo.out",
			delay: 1.2,
		});
	}, []);

	return (
		<>
			<section id="hero" className="relative overflow-hidden min-h-dvh flex flex-col justify-center py-20">
				<div className="noisy absolute inset-0 z-0 pointer-events-none"></div>

				<div className="relative z-10 flex flex-col items-center w-full px-5">
					<h1 className="title font-bold text-gradient leading-tight -translate-y-3 md:-translate-y-9">BluCare+</h1>

					<div className="hero-btn -mt-4 md:-mt-12 z-20">
						<div className="persistent-btn-glow">
							<button
								onClick={() => { window.location.href = import.meta.env.VITE_BACKEND_URL || 'http://127.0.0.1:8000'; }}
								className="badge flex items-center justify-center backdrop-blur-md bg-sage/10 border border-sage/20 text-sage px-8 py-3 rounded-full text-sm font-semibold uppercase tracking-widest hover:bg-sage hover:text-bg-base hover:scale-105 transition-all duration-300 transform cursor-pointer"
							>
								Begin Gently
							</button>
						</div>
					</div>
				</div>

				<img
					src="/images/hero-left-leaf.png"
					alt="left-leaf"
					className="left-leaf"
				/>
				<img
					src="/images/hero-right-leaf.png"
					alt="right-leaf"
					className="right-leaf"
				/>

				<div className="body relative md:absolute w-full px-5 mt-20 md:mt-0 md:bottom-20">

					<div className="content container mx-auto">
						<div className="flex flex-col lg:flex-row items-center lg:items-end justify-between gap-10 w-full">
							<div className="space-y-5 text-center lg:text-left">
								<p className="text-secondary uppercase tracking-[0.2em] font-medium text-xs">Clinical Intelligence Platform</p>
								<p className="subtitle font-medium text-xl md:text-2xl">
									Advanced Diagnostics <br className="hidden md:block" /> Gentle Human Care
								</p>
							</div>

							<div className="view-cocktails lg:max-w-xs text-center lg:text-left">
								<p className="subtitle text-sm md:text-base mb-6">
									BluCare+ harmonizes medical precision with empathetic understanding, providing a safe and intelligent space for your health journey.
								</p>
								<a href="#cocktails" className="group inline-flex items-center gap-3 text-sage font-medium hover:text-aqua transition-colors">
									<span>Explore Architecture</span>
									<div className="w-8 h-[1px] bg-sage group-hover:w-12 group-hover:bg-aqua transition-all"></div>
								</a>
							</div>
						</div>
					</div>
				</div>
			</section>

			<div className="video absolute inset-0 pointer-events-none opacity-30">
				<video
					ref={videoRef}
					muted
					playsInline
					preload="auto"
					src="/images/hero.mp4"
				/>
			</div>
		</>
	);
};

export default Hero;